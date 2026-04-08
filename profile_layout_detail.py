#!/usr/bin/env python3
"""Detailed per-step breakdown of layout_ext across all pages.

Measures the fixed pipeline (text_block-only mask transfer) on every page
and reports per-page + aggregate statistics.

Usage (inside container):
    python3 profile_layout_detail.py /root/input/img/*.jpg
"""

import glob
import os
import sys
import time

import cv2
import numpy as np
import torch
from lxml import etree as ET

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "submodules", "ndl_layout"))

from mmdet.apis import inference_detector, init_detector
import mmengine


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image paths")
    args = parser.parse_args()

    config = "submodules/ndl_layout/models/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco.py"
    checkpoint = "submodules/ndl_layout/models/ndl_retrainmodel.pth"

    print("Loading model...", flush=True)
    t0 = time.perf_counter()
    model = init_detector(config, checkpoint, "cuda:0")
    cfg = mmengine.Config.fromfile(config)
    classes = cfg.classes
    t_load = time.perf_counter() - t0
    print(f"Model loaded in {t_load:.1f}s  ({len(classes)} classes)")

    tb_cls_id = classes.index("text_block")

    from submodules.ndl_layout.tools.process_textblock import (
        textblock_to_polygon, refine_tb_polygons, get_relationship,
        convert_to_xml_string_with_data,
        gpu_refine_tb_masks, _masks_to_polygons,
    )

    # Warm up
    dummy = np.random.randint(0, 255, (2480, 3509, 3), dtype=np.uint8)
    _ = inference_detector(model, dummy)

    imgs = sorted(args.images)
    print(f"\nProcessing {len(imgs)} images...\n")

    # Collect per-page timings
    records = []

    for img_path in imgs:
        page_name = os.path.basename(img_path)

        # --- imread ---
        t0 = time.perf_counter()
        img = cv2.imread(img_path)
        t_imread = time.perf_counter() - t0
        img_h, img_w = img.shape[:2]

        # --- GPU inference ---
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = inference_detector(model, img)
        torch.cuda.synchronize()
        t_infer = time.perf_counter() - t0

        n_inst = len(result.pred_instances.bboxes)
        n_tb = int((result.pred_instances.labels == tb_cls_id).sum())

        # --- OLD CPU PATH: bbox+mask transfer (text_block only) ---
        t0 = time.perf_counter()
        res_bbox = {i: [] for i in range(len(classes))}
        res_segm = {i: [] for i in range(len(classes))}
        for bbox, segm, cls, score in zip(
            result.pred_instances.bboxes,
            result.pred_instances.masks,
            result.pred_instances.labels,
            result.pred_instances.scores,
        ):
            cls = int(cls)
            s_bbox = bbox.to("cpu").detach().numpy().copy().tolist()
            s_bbox.append(float(score))
            res_bbox[cls].append(s_bbox)
            if cls == tb_cls_id:
                res_segm[cls].append(segm.to("cpu").detach().numpy().copy())
        t_transfer = time.perf_counter() - t0

        # --- OLD: textblock_to_polygon ---
        t0 = time.perf_counter()
        tb_polygons = textblock_to_polygon(classes, res_segm, min_bbox_size=5)
        t_polygon = time.perf_counter() - t0

        # --- OLD: refine_tb_polygons ---
        t0 = time.perf_counter()
        tb_polygons = refine_tb_polygons(tb_polygons)
        t_refine = time.perf_counter() - t0

        # --- OLD: get_relationship ---
        t0 = time.perf_counter()
        tb_info, ad_info, independ_lines = get_relationship(
            res_bbox, tb_polygons, classes)
        t_rel = time.perf_counter() - t0

        # --- NEW GPU PATH ---
        # (a) GPU overlap removal
        tb_global_indices = []
        all_labels = result.pred_instances.labels.cpu().numpy()
        for idx in range(len(all_labels)):
            if int(all_labels[idx]) == tb_cls_id:
                tb_global_indices.append(idx)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if tb_global_indices:
            tb_masks_gpu = result.pred_instances.masks[tb_global_indices]
            survived = gpu_refine_tb_masks(tb_masks_gpu)
        else:
            survived = torch.ones(0, dtype=torch.bool, device="cuda:0")
        torch.cuda.synchronize()
        t_gpu_refine = time.perf_counter() - t0
        n_survived = int(survived.sum()) if len(survived) > 0 else 0

        # (b) Transfer surviving masks only
        t0 = time.perf_counter()
        survived_list = survived.cpu().tolist() if len(survived) > 0 else []
        surviving_masks = []
        for i in range(len(tb_global_indices)):
            if survived_list[i]:
                surviving_masks.append(
                    result.pred_instances.masks[tb_global_indices[i]].cpu().numpy())
        t_transfer_new = time.perf_counter() - t0

        # (c) Polygon on surviving masks only
        t0 = time.perf_counter()
        _ = _masks_to_polygons(surviving_masks, min_bbox_size=5)
        t_polygon_new = time.perf_counter() - t0

        # --- E2E: full convert_to_xml_string_with_data (uses GPU path automatically) ---
        t0 = time.perf_counter()
        xml_str = convert_to_xml_string_with_data(
            img_w, img_h, img_path, classes, result, score_thr=0.3)
        t_convert_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        node = ET.fromstring(
            '<?xml version="1.0" standalone="yes"?><OCRDATASET xmlns="">\n</OCRDATASET>\n')
        result_xml = ET.fromstring(xml_str)
        node.append(result_xml)
        tree = ET.ElementTree(node)
        t_lxml = time.perf_counter() - t0

        rec = {
            "page": page_name,
            "n_inst": n_inst,
            "n_tb": n_tb,
            "n_survived": n_survived,
            "imread": t_imread,
            "infer": t_infer,
            # Old CPU path steps
            "transfer": t_transfer,
            "polygon": t_polygon,
            "refine": t_refine,
            "rel": t_rel,
            # New GPU path steps
            "gpu_refine": t_gpu_refine,
            "transfer_new": t_transfer_new,
            "polygon_new": t_polygon_new,
            # E2E
            "convert_full": t_convert_full,
            "lxml": t_lxml,
        }
        records.append(rec)

        total = t_imread + t_infer + t_convert_full + t_lxml
        print(f"  {page_name}: {n_inst:3d} inst, {n_tb:2d} tb ({n_survived} survived) | "
              f"imread={t_imread:.3f} infer={t_infer:.3f} "
              f"convert={t_convert_full:.3f} lxml={t_lxml:.3f} | "
              f"total={total:.3f}s")

    # --- Aggregate ---
    n = len(records)
    print(f"\n{'='*70}")
    print(f"{'Step':20s} {'Sum':>8s} {'Avg':>8s} {'Min':>8s} {'Max':>8s} {'%':>6s}")
    print(f"{'='*70}")

    steps = ["imread", "infer", "transfer", "polygon", "refine", "rel", "convert_full", "lxml"]
    totals = {}
    for step in steps:
        vals = [r[step] for r in records]
        totals[step] = sum(vals)

    grand = sum(totals[s] for s in ["imread", "infer", "convert_full", "lxml"])

    for step in steps:
        vals = [r[step] for r in records]
        s = sum(vals)
        if step in ("transfer", "polygon", "refine", "rel"):
            pct = s / totals["convert_full"] * 100 if totals["convert_full"] > 0 else 0
            label = f"{step} (of convert)"
        else:
            pct = s / grand * 100 if grand > 0 else 0
            label = step
        print(f"  {label:20s} {s:8.2f}s {s/n:8.3f}s {min(vals):8.3f}s {max(vals):8.3f}s {pct:5.1f}%")

    print(f"{'='*70}")
    print(f"  {'TOTAL':20s} {grand:8.2f}s {grand/n:8.3f}s")

    # GPU path comparison
    gpu_steps = ["gpu_refine", "transfer_new", "polygon_new"]
    print(f"\n{'='*70}")
    print(f"  GPU PATH COMPARISON (individual steps)")
    print(f"{'='*70}")
    old_sum = totals["transfer"] + totals["polygon"] + totals["refine"]
    new_sum = 0
    for step in gpu_steps:
        vals = [r[step] for r in records]
        s = sum(vals)
        new_sum += s
        print(f"  {step:20s} {s:8.2f}s {s/n:8.3f}s {min(vals):8.3f}s {max(vals):8.3f}s")
    print(f"  {'OLD total':20s} {old_sum:8.2f}s {old_sum/n:8.3f}s")
    print(f"  {'NEW total':20s} {new_sum:8.2f}s {new_sum/n:8.3f}s")
    if old_sum > 0:
        print(f"  Speedup: {old_sum/new_sum:.1f}x ({(1 - new_sum/old_sum)*100:.0f}% reduction)")

    # Instance stats
    inst_vals = [r["n_inst"] for r in records]
    tb_vals = [r["n_tb"] for r in records]
    surv_vals = [r["n_survived"] for r in records]
    print(f"\n  Instances: avg={sum(inst_vals)/n:.0f}, min={min(inst_vals)}, max={max(inst_vals)}")
    print(f"  text_block: avg={sum(tb_vals)/n:.0f}, min={min(tb_vals)}, max={max(tb_vals)}")
    print(f"  survived: avg={sum(surv_vals)/n:.0f}, min={min(surv_vals)}, max={max(surv_vals)}")


if __name__ == "__main__":
    main()
