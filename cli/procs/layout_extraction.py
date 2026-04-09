# Copyright (c) 2023, National Diet Library, Japan
#
# This software is released under the CC BY 4.0.
# https://creativecommons.org/licenses/by/4.0/


import os
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor

import lxml
import lxml.etree
import numpy
from PIL import Image

from mmdet.apis import inference_detector
from submodules.ndl_layout.tools.process_textblock import convert_to_xml_string_with_data

from .base_proc import BaseInferenceProcess


class LayoutExtractionProcess(BaseInferenceProcess):
    """
    レイアウト抽出推論を実行するプロセスのクラス。
    BaseInferenceProcessを継承しています。
    """
    def __init__(self, cfg, pid):
        """
        Parameters
        ----------
        cfg : dict
            本実行処理における設定情報です。
        pid : int
            実行される順序を表す数値。
        """
        super().__init__(cfg, pid, '_layer_ext')
        from submodules.ndl_layout.tools.process_textblock import InferencerWithCLI
        self._inferencer = InferencerWithCLI(self.cfg['layout_extraction'])
        self._run_submodule_inference = self._inferencer.inference_with_cli

    def is_valid_input(self, input_data):
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
            print('LayoutExtractionProcess: input img is not numpy.ndarray')
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
        print('### Layout Extraction Process ###')
        output_data = input_data.copy()
        inference_output = self._run_submodule_inference(
            img=input_data['img'],
            img_path=input_data['img_file_name'],
            score_thr=self.cfg['layout_extraction']['score_thr'],
            dump=(self.cfg['dump'] or self.cfg['save_image'])
        )

        # Create result to pass xml and img data
        result = []
        output_data['xml'] = ET.ElementTree(
            ET.fromstring(lxml.etree.tostring(inference_output['xml']))
        )
        if inference_output['dump_img'] is not None:
            output_data['dump_img'] = inference_output['dump_img']
        result.append(output_data)
        return result

    def do_batch(self, items, crop_params=None, gpu_sub_batch=4):
        """バッチ GPU 推論 + per-item CPU 後処理でレイアウト抽出を実行する。
        crop_params が渡された場合、CPU 後処理と line_ocr の crop 収集を
        ThreadPoolExecutor で並列実行する。
        """
        imgs = [item['img'] for item in items]
        score_thr = self.cfg['layout_extraction']['score_thr']
        classes = self._inferencer.detector.classes

        # GPU 推論をサブバッチで実行 (VRAM 制約)
        all_results = []
        for i in range(0, len(imgs), gpu_sub_batch):
            sub_batch = imgs[i:i + gpu_sub_batch]
            sub_results = inference_detector(self._inferencer.detector.model, sub_batch)
            if not isinstance(sub_results, list):
                sub_results = [sub_results]
            all_results.extend(sub_results)

        # CPU 後処理 (+ crop 収集)
        if crop_params:
            return self._cpu_postprocess_threaded(items, all_results, classes, score_thr, crop_params)
        else:
            return self._cpu_postprocess_sequential(items, all_results, classes, score_thr)

    def _cpu_postprocess_sequential(self, items, all_results, classes, score_thr):
        """逐次 CPU 後処理 (フォールバック)。"""
        output_items = []
        for item, result in zip(items, all_results):
            output_data = item.copy()
            output_data['xml'] = self._build_xml(item, result, classes, score_thr)
            output_items.append(output_data)
        return output_items

    def _cpu_postprocess_threaded(self, items, all_results, classes, score_thr, crop_params):
        """ThreadPoolExecutor で XML 変換 + crop 収集を並列実行する。"""
        from submodules.text_recognition_lightning.src.datamodules.ndl_components.ndl_dataset import (
            XMLRawDatasetWithCli, XMLRawAttrWithCli,
        )

        transforms = crop_params['transforms']
        batch_max_length = crop_params['batch_max_length']
        additional_elements = crop_params['additional_elements']

        def process_page(item, result, xml_tree):
            pil_image = Image.fromarray(item['img'])
            pid = os.path.basename(
                item.get('img_path', item.get('img_file_name', 'x'))
            ).split('_')[0]

            crop_ds = XMLRawDatasetWithCli(
                transforms=transforms,
                batch_max_length=batch_max_length,
                additional_elements=additional_elements,
            )
            crop_ds.set_data(pil_image, xml_tree, pid)
            tensors = [t for t, _, _ in crop_ds]

            attr_iter = XMLRawAttrWithCli(
                item, additional_elements=additional_elements,
            )
            attr_iter.set_data(xml_tree, pid)
            line_elems = list(attr_iter)

            return tensors, line_elems

        # lxml はスレッドセーフでないため XML 構築はメインスレッドで実行
        xml_trees = [self._build_xml(item, result, classes, score_thr)
                     for item, result in zip(items, all_results)]

        output_items = []
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(process_page, item, result, xml_tree)
                for item, result, xml_tree in zip(items, all_results, xml_trees)
            ]
            for item, xml_tree, future in zip(items, xml_trees, futures):
                tensors, line_elems = future.result()
                output_data = item.copy()
                output_data['xml'] = xml_tree
                output_data['_line_tensors'] = tensors
                output_data['_line_elements'] = line_elems
                output_items.append(output_data)

        return output_items

    @staticmethod
    def _build_xml(item, result, classes, score_thr):
        """mmdet 結果から stdlib ElementTree を構築する。"""
        img = item['img']
        xml_str = convert_to_xml_string_with_data(
            img.shape[1], img.shape[0], item['img_file_name'],
            classes, result, score_thr=score_thr)
        result_xml = lxml.etree.fromstring(xml_str)
        node = lxml.etree.fromstring(
            '<?xml version="1.0" standalone="yes"?>'
            '<OCRDATASET xmlns="">\n</OCRDATASET>\n')
        node.append(result_xml)
        return ET.ElementTree(ET.fromstring(lxml.etree.tostring(node)))
