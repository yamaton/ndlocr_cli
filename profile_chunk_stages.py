#!/usr/bin/env python3
"""Profile per-stage wall time in the chunked pipeline.

Uses the actual production do_batch() paths (no monkey-patching).
Measures GPU-synchronized wall time per stage per chunk.

Usage (inside container):
    python3 profile_chunk_stages.py /root/input /root/out
    python3 profile_chunk_stages.py /root/input /root/out --chunk-size 6
"""

import argparse
import os
import time

import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_DIR)


def sync_time():
    """GPU-synchronized wall time."""
    torch.cuda.synchronize()
    return time.perf_counter()


def profile_pipeline(input_root, output_root, chunk_size):
    from cli.core import utils
    from cli.core.inference import OcrInferrer

    cfg_dict = {
        'input_root': input_root,
        'output_root': output_root,
        'config_file': 'config.yml',
        'proc_range': '0..3',
        'save_image': False,
        'save_xml': True,
        'dump': False,
        'input_structure': 's',
        'ruby_only': False,
    }

    infer_cfg = utils.parse_cfg(cfg_dict)
    infer_cfg['pipeline']['chunk_size'] = chunk_size
    infer_cfg['output_root'] = utils.mkdir_with_duplication_check(infer_cfg['output_root'])

    inferrer = OcrInferrer(infer_cfg)

    # Get input data
    single_outputdir_data_list = inferrer._get_single_dir_data(infer_cfg['input_dirs'][0])
    if not single_outputdir_data_list:
        print("No input data found")
        return

    single_outputdir_data = single_outputdir_data_list[0]
    img_list = single_outputdir_data['img_list']
    print(f"Pages: {len(img_list)}, chunk_size: {chunk_size}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # Warm-up: first chunk (uses real do_batch paths)
    warmup_paths = img_list[:chunk_size]
    warmup_items = []
    for img_path in warmup_paths:
        item_data = inferrer._get_single_image_file_data(img_path, single_outputdir_data)
        if item_data:
            warmup_items.extend(item_data)
    if warmup_items:
        torch.cuda.empty_cache()
        for proc in inferrer.proc_list:
            warmup_items = inferrer._run_proc_on_chunk(proc, warmup_items)
        print("Warm-up chunk done\n")

    # Measured chunks — using real production code paths
    chunk_timings = []  # (chunk_idx, stage_name, wall_time)

    for chunk_start in range(0, len(img_list), chunk_size):
        chunk_paths = img_list[chunk_start:chunk_start + chunk_size]
        items = []
        for img_path in chunk_paths:
            item_data = inferrer._get_single_image_file_data(img_path, single_outputdir_data)
            if item_data:
                items.extend(item_data)
        if not items:
            continue

        chunk_idx = chunk_start // chunk_size
        torch.cuda.empty_cache()
        t_chunk_start = sync_time()

        for proc in inferrer.proc_list:
            t_stage_start = sync_time()
            items = inferrer._run_proc_on_chunk(proc, items)
            t_stage_end = sync_time()
            chunk_timings.append((chunk_idx, proc.proc_name, t_stage_end - t_stage_start))

        t_chunk_end = sync_time()
        chunk_timings.append((chunk_idx, 'TOTAL', t_chunk_end - t_chunk_start))

    # --- Report ---
    print("=" * 70)
    print("PER-CHUNK WALL TIME (production code paths)")
    print("=" * 70)
    chunk_ids = sorted(set(c[0] for c in chunk_timings))
    for cid in chunk_ids:
        entries = [(name, t) for (i, name, t) in chunk_timings if i == cid]
        parts = [f"{name}={t:.3f}s" for name, t in entries]
        print(f"  chunk {cid}: {', '.join(parts)}")

    # Per-stage averages
    print("\n" + "=" * 70)
    print("PER-STAGE AVERAGES")
    print("=" * 70)
    stage_names = []
    for _, name, _ in chunk_timings:
        if name not in stage_names:
            stage_names.append(name)
    for name in stage_names:
        times = [t for (_, n, t) in chunk_timings if n == name]
        avg = sum(times) / len(times)
        total = sum(times)
        print(f"  {name:25s}  avg={avg:.3f}s  total={total:.1f}s  ({len(times)} chunks)")

    # Total wall time
    total_wall = sum(t for (_, name, t) in chunk_timings if name == 'TOTAL')
    n_pages = len(img_list)
    print(f"\n  Wall time: {total_wall:.1f}s  ({total_wall/n_pages:.2f}s/page, {n_pages} pages)")


def main():
    parser = argparse.ArgumentParser(description="Profile chunked pipeline per-stage wall time")
    parser.add_argument("input_root")
    parser.add_argument("output_root")
    parser.add_argument("--chunk-size", type=int, default=6)
    args = parser.parse_args()

    profile_pipeline(args.input_root, args.output_root, args.chunk_size)


if __name__ == "__main__":
    main()
