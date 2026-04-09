# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/
import copy
import os

import hydra
import numpy
import torch
import xml.etree.ElementTree as ET
from PIL import Image

from .base_proc import BaseInferenceProcess


class LineOcrProcess(BaseInferenceProcess):
    """
    行文字認識推論を実行するプロセスのクラス。
    BaseInferenceProcessを継承しています。
    """
    def __init__(self, cfg, pid):
        """
        Parameters
        ----------
        cfg : dict
            本推論処理における設定情報です。
        pid : int
            実行される順序を表す数値。
        """
        super().__init__(cfg, pid, '_line_ocr')
        from submodules.text_recognition_lightning.src.tasks.infer_task import infer, create_object_dict
        self._run_submodule_inference = infer

        config_path = "../../submodules/text_recognition_lightning/configs"
        hydra.initialize(version_base="1.2", config_path=config_path)
        self._hydra_cfg = hydra.compose(config_name="infer", overrides=[f"paths.output_dir={cfg['output_root']}"])
        self._hydra_cfg['model']['character_file'] = cfg['line_ocr']['char_list']
        self._hydra_cfg['ckpt_path'] = cfg['line_ocr']['saved_model']
        self._hydra_cfg = self._remove_noise_elements(self._hydra_cfg)
        for element_type, add_flag in cfg['line_ocr']['additional_elements'].items():
            if add_flag:
                add_block_string = f'BLOCK[@TYPE="{element_type}"]'
                self._hydra_cfg['datamodule']['additional_elements'].append(add_block_string)
        from pathlib import Path
        hydra.core.utils._save_config(self._hydra_cfg, "config.yaml", Path(cfg['output_root'])/".text_recognition")

        self._object_dict = create_object_dict(self._hydra_cfg)

    def _remove_noise_elements(self, hydra_cfg):
        NOISE_ELEMENT_TYPE = ['ノンブル', '柱']

        for element_type in NOISE_ELEMENT_TYPE:
            for idx, value in enumerate(hydra_cfg['datamodule']['additional_elements']):
                if value == f'BLOCK[@TYPE="{element_type}"]':
                    hydra_cfg['datamodule']['additional_elements'].pop(idx)
                    break
        return hydra_cfg

    def _is_valid_input(self, input_data):
        """
        本クラスの推論処理における入力データのバリデーション。

        Parameters
        ----------
        input_data : dict
            推論処理を実行する対象の入力データ。

        Returns
        -------
        [変数なし] : bool
            入力データが正しければTrue, そうでなければFalseを返します。
        """
        if type(input_data['img']) is not numpy.ndarray:
            print('LineOcrProcess: input img is not numpy.ndarray')
            return False
        if type(input_data['xml']) is not ET.ElementTree:
            print('LineOcrProcess: input xml is not ElementTree')
            return False
        return True

    def _run_process(self, input_data):
        """
        推論処理の本体部分。

        Parameters
        ----------
        input_data : dict
            推論処理を実行する対象の入力データ。

        Returns
        -------
        result : dict
            推論処理の結果を保持する辞書型データ。
            基本的にinput_dataと同じ構造です。
        """
        result = []

        print('### Line OCR Process ###')
        output_data = self._run_submodule_inference(self._object_dict, input_data)
        result.append(output_data)

        return result

    def do_batch(self, items):
        """全ページの LINE crop を一括収集し、Trainer を迂回して直接 GPU 推論する。"""
        from submodules.text_recognition_lightning.src.datamodules.ndl_components.ndl_dataset import (
            XMLRawDatasetWithCli, XMLRawAttrWithCli,
        )

        cfg = self._hydra_cfg
        model = self._object_dict['model']
        datamodule = self._object_dict['datamodule']

        # model を GPU に配置 (初回のみ移動)
        device = next(model.parameters()).device
        if str(device) == 'cpu':
            model.cuda()
        model.eval()

        output_items = []
        all_tensors = []
        line_metadata = []       # (page_idx, line_idx)
        page_line_elements = []  # per-page LINE 要素参照リスト

        # Phase A: 全ページから LINE crop を収集
        for page_idx, item in enumerate(items):
            output_data = item.copy()
            output_data['xml'] = copy.deepcopy(item['xml'])
            output_items.append(output_data)

            pil_image = Image.fromarray(item['img'])
            pid = os.path.basename(item.get('img_path', item.get('img_file_name', 'unknown'))).split('_')[0]

            # crop 用 dataset (transforms 付き)
            crop_dataset = XMLRawDatasetWithCli(
                transforms=datamodule.transforms_test,
                batch_max_length=cfg.datamodule.test_batch_max_length,
                additional_elements=cfg.datamodule.additional_elements,
            )
            crop_dataset.set_data(pil_image, item['xml'], pid)

            line_idx = 0
            for tensor, label, meta in crop_dataset:
                all_tensors.append(tensor)
                line_metadata.append((page_idx, line_idx))
                line_idx += 1

            # 書き戻し用 LINE 要素参照 (同じ反復順序)
            attr_iter = XMLRawAttrWithCli(
                output_data,
                additional_elements=cfg.datamodule.additional_elements,
            )
            attr_iter.set_data(output_data['xml'], pid)
            page_line_elements.append(list(attr_iter))

        if not all_tensors:
            return output_items

        # 反復順序の安全ガード
        from collections import Counter
        crop_counts = Counter(pi for pi, _ in line_metadata)
        for page_idx, elems in enumerate(page_line_elements):
            assert len(elems) == crop_counts.get(page_idx, 0), (
                f"page {page_idx}: LINE element count mismatch: "
                f"crops={crop_counts.get(page_idx, 0)}, attrs={len(elems)}"
            )

        # Phase B: バッチ GPU 推論
        batch_size = cfg.datamodule.batch_size  # 128
        all_texts = []

        with torch.no_grad():
            for i in range(0, len(all_tensors), batch_size):
                batch = torch.stack(all_tensors[i:i + batch_size]).cuda()
                preds = model(batch)
                if isinstance(preds, tuple):
                    preds = preds[0]
                bs = preds.size(0)
                preds_size = torch.full((bs,), preds.size(1), dtype=torch.int32)
                preds_index = preds.argmax(2)
                texts = model.converter.decode(preds_index.data, preds_size.data)
                all_texts.extend(texts)

        del all_tensors  # CPU RAM 早期解放

        # Phase C: 結果を LINE 要素に書き戻し
        for (page_idx, line_idx), text in zip(line_metadata, all_texts):
            line_elem = page_line_elements[page_idx][line_idx]
            line_elem.attrib['STRING'] = text

        return output_items
