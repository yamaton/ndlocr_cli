#!/usr/bin/env python3
"""Regression test: compare CPU refine_tb_polygons vs GPU gpu_refine_tb_masks.

For each page, runs both paths and compares which text_block masks survive.
Also supports threshold sweep to find the optimal containment threshold.

Usage (inside container):
    python3 test_gpu_refine.py /root/input/img/*.jpg
    python3 test_gpu_refine.py --sweep /root/input/img/*.jpg
"""

import os
import sys

import cv2
import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)
sys.path.insert(0, os.path.join(PROJECT_DIR, "submodules", "ndl_layout"))

from mmdet.apis import inference_detector, init_detector
import mmengine

from submodules.ndl_layout.tools.process_textblock import (
    textblock_to_polygon, refine_tb_polygons,
    gpu_refine_tb_masks,
)


def cpu_path(result, classes, tb_cls_id, min_bbox_size=5):
    """Run original CPU path: transfer all → polygon → refine. Return survived set."""
    res_segm = {i: [] for i in range(len(classes))}
    for segm, cls in zip(result.pred_instances.masks, result.pred_instances.labels):
        cls = int(cls)
        if cls == tb_cls_id:
            res_segm[cls].append(segm.to("cpu").detach().numpy().copy())

    tb_polygons = textblock_to_polygon(classes, res_segm, min_bbox_size)
    tb_polygons_refined = refine_tb_polygons(tb_polygons)
    survived = set(i for i, p in enumerate(tb_polygons_refined) if p is not None)
    return survived, tb_polygons_refined


def gpu_path(result, classes, tb_cls_id, threshold=0.95):
    """Run GPU path: overlap removal on GPU. Return survived set."""
    all_labels = result.pred_instances.labels.cpu().numpy()
    tb_global_indices = [i for i, l in enumerate(all_labels) if int(l) == tb_cls_id]

    if not tb_global_indices:
        return set(), []

    tb_masks_gpu = result.pred_instances.masks[tb_global_indices]
    survived_tensor = gpu_refine_tb_masks(tb_masks_gpu, threshold=threshold)
    survived_list = survived_tensor.cpu().tolist()
    survived = set(i for i, s in enumerate(survived_list) if s)
    return survived, survived_list


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image paths")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep threshold from 0.85 to 0.99")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Containment threshold (default: 0.95)")
    args = parser.parse_args()

    config = "submodules/ndl_layout/models/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco.py"
    checkpoint = "submodules/ndl_layout/models/ndl_retrainmodel.pth"

    print("Loading model...", flush=True)
    model = init_detector(config, checkpoint, "cuda:0")
    cfg = mmengine.Config.fromfile(config)
    classes = cfg.classes
    tb_cls_id = classes.index("text_block")

    # Warm up
    dummy = np.random.randint(0, 255, (2480, 3509, 3), dtype=np.uint8)
    _ = inference_detector(model, dummy)

    imgs = sorted(args.images)
    print(f"Processing {len(imgs)} images...\n")

    # Cache inference results (discard images to save memory)
    results = []
    for img_path in imgs:
        img = cv2.imread(img_path)
        torch.cuda.synchronize()
        result = inference_detector(model, img)
        torch.cuda.synchronize()
        results.append((img_path, result))

    if args.sweep:
        # Threshold sweep
        thresholds = [round(0.85 + i * 0.01, 2) for i in range(15)]
        print(f"{'Threshold':>10s} {'Match':>6s} {'Disagree':>8s} {'Extra_GPU':>10s} {'Miss_GPU':>10s}")
        print("-" * 50)

        for thr in thresholds:
            n_match = 0
            n_disagree = 0
            n_extra_gpu = 0  # GPU keeps but CPU removes
            n_miss_gpu = 0   # GPU removes but CPU keeps

            for img_path, result in results:
                cpu_survived, _ = cpu_path(result, classes, tb_cls_id)
                gpu_survived, _ = gpu_path(result, classes, tb_cls_id, threshold=thr)

                if cpu_survived == gpu_survived:
                    n_match += 1
                else:
                    n_disagree += 1
                    n_extra_gpu += len(gpu_survived - cpu_survived)
                    n_miss_gpu += len(cpu_survived - gpu_survived)

            marker = " <--" if n_disagree == 0 else ""
            print(f"  {thr:8.2f} {n_match:6d} {n_disagree:8d} {n_extra_gpu:10d} {n_miss_gpu:10d}{marker}")

    else:
        # Single threshold comparison
        n_match = 0
        n_disagree = 0
        disagree_details = []

        for img_path, result in results:
            page_name = os.path.basename(img_path)
            n_tb = int((result.pred_instances.labels == tb_cls_id).sum())

            cpu_survived, _ = cpu_path(result, classes, tb_cls_id)
            gpu_survived, _ = gpu_path(result, classes, tb_cls_id, threshold=args.threshold)

            if cpu_survived == gpu_survived:
                n_match += 1
                status = "OK"
            else:
                n_disagree += 1
                status = "DIFF"
                extra_gpu = gpu_survived - cpu_survived
                miss_gpu = cpu_survived - gpu_survived

                # Compute overlap ratios for disagreed masks
                all_labels = result.pred_instances.labels.cpu().numpy()
                tb_global = [i for i, l in enumerate(all_labels) if int(l) == tb_cls_id]
                tb_masks_gpu = result.pred_instances.masks[tb_global]
                areas = tb_masks_gpu.view(len(tb_global), -1).sum(dim=1).float()

                detail = {"page": page_name, "extra_gpu": [], "miss_gpu": []}
                for idx in extra_gpu:
                    inter = (tb_masks_gpu[idx] & tb_masks_gpu).view(len(tb_global), -1).sum(dim=1).float()
                    ratio = inter / (areas[idx] + 1e-6)
                    ratio[idx] = 0.0
                    detail["extra_gpu"].append((idx, float(ratio.max())))

                for idx in miss_gpu:
                    inter = (tb_masks_gpu[idx] & tb_masks_gpu).view(len(tb_global), -1).sum(dim=1).float()
                    ratio = inter / (areas[idx] + 1e-6)
                    ratio[idx] = 0.0
                    detail["miss_gpu"].append((idx, float(ratio.max())))

                disagree_details.append(detail)

            print(f"  {page_name}: {n_tb:2d} tb | CPU={len(cpu_survived)} GPU={len(gpu_survived)} | {status}")

        print(f"\n{'='*60}")
        print(f"  Match: {n_match}/{len(imgs)}, Disagree: {n_disagree}/{len(imgs)}")
        print(f"  Threshold: {args.threshold}")

        if disagree_details:
            print(f"\nDisagreement details:")
            for d in disagree_details:
                print(f"  {d['page']}:")
                for idx, ratio in d["extra_gpu"]:
                    print(f"    GPU keeps #{idx}: max_overlap_ratio={ratio:.4f}")
                for idx, ratio in d["miss_gpu"]:
                    print(f"    GPU removes #{idx}: max_overlap_ratio={ratio:.4f}")


if __name__ == "__main__":
    main()
