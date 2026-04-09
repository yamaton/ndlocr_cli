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

    def do_batch(self, items, **kwargs):
        """全ページの LINE crop を一括収集し、Trainer を迂回して直接 GPU 推論する。
        layout_ext が _line_tensors を事前計算済みならそれを使う。
        """
        if items and '_line_tensors' in items[0]:
            return self._do_batch_precomputed(items)
        return self._do_batch_full(items)

    def _ensure_model_on_gpu(self):
        """model を GPU に配置 (初回のみ)。"""
        model = self._object_dict['model']
        device = next(model.parameters()).device
        if str(device) == 'cpu':
            model.cuda()
        model.eval()
        return model

    def _run_gpu_inference(self, model, all_tensors):
        """バッチ GPU 推論を実行し、テキストリストを返す。"""
        batch_size = self._hydra_cfg.datamodule.batch_size  # 128
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

        return all_texts

    def _do_batch_precomputed(self, items):
        """layout_ext が事前計算した _line_tensors / _line_elements を使って GPU 推論のみ実行。"""
        model = self._ensure_model_on_gpu()

        all_tensors = []
        page_line_elements = []

        for item in items:
            all_tensors.extend(item['_line_tensors'])
            page_line_elements.append(item['_line_elements'])

        if not all_tensors:
            return items

        # 安全ガード
        for page_idx, elems in enumerate(page_line_elements):
            n_tensors = len(items[page_idx]['_line_tensors'])
            assert len(elems) == n_tensors, (
                f"page {page_idx}: LINE element count mismatch: "
                f"tensors={n_tensors}, attrs={len(elems)}"
            )

        all_texts = self._run_gpu_inference(model, all_tensors)
        del all_tensors

        # 結果を LINE 要素に書き戻し
        text_idx = 0
        for page_idx, elems in enumerate(page_line_elements):
            for line_elem in elems:
                line_elem.attrib['STRING'] = all_texts[text_idx]
                text_idx += 1

        # 一時データを削除
        for item in items:
            del item['_line_tensors']
            del item['_line_elements']

        return items

    def _do_batch_full(self, items):
        """従来パス: crop 収集 + GPU 推論 + 書き戻し。"""
        from submodules.text_recognition_lightning.src.datamodules.ndl_components.ndl_dataset import (
            XMLRawDatasetWithCli, XMLRawAttrWithCli,
        )

        cfg = self._hydra_cfg
        model = self._ensure_model_on_gpu()
        datamodule = self._object_dict['datamodule']

        output_items = []
        all_tensors = []
        line_metadata = []
        page_line_elements = []

        for page_idx, item in enumerate(items):
            output_data = item.copy()
            output_data['xml'] = copy.deepcopy(item['xml'])
            output_items.append(output_data)

            pil_image = Image.fromarray(item['img'])
            pid = os.path.basename(item.get('img_path', item.get('img_file_name', 'unknown'))).split('_')[0]

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

            attr_iter = XMLRawAttrWithCli(
                output_data,
                additional_elements=cfg.datamodule.additional_elements,
            )
            attr_iter.set_data(output_data['xml'], pid)
            page_line_elements.append(list(attr_iter))

        if not all_tensors:
            return output_items

        from collections import Counter
        crop_counts = Counter(pi for pi, _ in line_metadata)
        for page_idx, elems in enumerate(page_line_elements):
            assert len(elems) == crop_counts.get(page_idx, 0), (
                f"page {page_idx}: LINE element count mismatch: "
                f"crops={crop_counts.get(page_idx, 0)}, attrs={len(elems)}"
            )

        all_texts = self._run_gpu_inference(model, all_tensors)
        del all_tensors

        for (page_idx, line_idx), text in zip(line_metadata, all_texts):
            page_line_elements[page_idx][line_idx].attrib['STRING'] = text

        return output_items
