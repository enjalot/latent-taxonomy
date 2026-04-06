"""
Benchmark SAE encode throughput on CPU vs GPU (MPS on Mac, CUDA on Linux).

Tests different batch sizes to find optimal throughput for feature extraction.
The key question: for a small SAE (384-dim input, 3072 features), is GPU worth
the data transfer overhead?

Usage:
    python -m pipeline.benchmark_sae --model-path ~/code/latent-sae/checkpoints/sae
    python -m pipeline.benchmark_sae --model-repo enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT --model-name 64_32
"""

import argparse
import time
import sys

import numpy as np
import torch


def load_model(args):
    from latentsae.sae import Sae

    if args.model_path:
        import os
        path = os.path.expanduser(args.model_path)
        print(f"Loading from disk: {path}")
        return Sae.load_from_disk(path, device="cpu")
    else:
        print(f"Loading from hub: {args.model_repo} / {args.model_name}")
        return Sae.load_from_hub(args.model_repo, args.model_name, device="cpu")


def benchmark_encode(model, device, d_in, n_samples, batch_size, warmup=3):
    """Benchmark encode throughput on a specific device."""
    model_on_device = model.to(device)
    model_on_device.eval()

    # Generate random embeddings (simulates real data)
    data = torch.randn(n_samples, d_in, dtype=torch.float32)

    num_batches = (n_samples + batch_size - 1) // batch_size

    # Warmup
    with torch.no_grad():
        for i in range(min(warmup, num_batches)):
            s = i * batch_size
            e = min(s + batch_size, n_samples)
            batch = data[s:e].to(device)
            _ = model_on_device.encode(batch)

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    # Timed run
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(num_batches):
            s = i * batch_size
            e = min(s + batch_size, n_samples)
            batch = data[s:e].to(device)
            out = model_on_device.encode(batch)
            # Force sync for accurate timing
            _ = out.top_acts.cpu()
            _ = out.top_indices.cpu()

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - t0
    throughput = n_samples / elapsed

    # Move model back to CPU to free device memory
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return elapsed, throughput


def benchmark_encode_no_transfer(model, device, d_in, n_samples, batch_size, warmup=3):
    """Benchmark encode with data already on device (no CPU→GPU transfer)."""
    model_on_device = model.to(device)
    model_on_device.eval()

    # Pre-allocate on device
    data = torch.randn(n_samples, d_in, dtype=torch.float32, device=device)
    k = model_on_device.cfg.k
    all_acts = torch.zeros(n_samples, k, dtype=torch.float32, device=device)
    all_indices = torch.zeros(n_samples, k, dtype=torch.int64, device=device)

    num_batches = (n_samples + batch_size - 1) // batch_size

    # Warmup
    with torch.no_grad():
        for i in range(min(warmup, num_batches)):
            s = i * batch_size
            e = min(s + batch_size, n_samples)
            _ = model_on_device.encode(data[s:e])

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    # Timed run
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(num_batches):
            s = i * batch_size
            e = min(s + batch_size, n_samples)
            out = model_on_device.encode(data[s:e])
            all_acts[s:e] = out.top_acts
            all_indices[s:e] = out.top_indices

    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - t0
    throughput = n_samples / elapsed

    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return elapsed, throughput


def main():
    parser = argparse.ArgumentParser(description="Benchmark SAE encode throughput")
    parser.add_argument("--model-path", default=None, help="Local SAE checkpoint directory")
    parser.add_argument("--model-repo", default=None, help="HuggingFace repo")
    parser.add_argument("--model-name", default="64_32", help="Model name in repo")
    parser.add_argument("--n-samples", type=int, default=100_000, help="Number of samples to encode")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                        help="Batch sizes to test")
    args = parser.parse_args()

    if not args.model_path and not args.model_repo:
        print("Error: provide --model-path or --model-repo")
        sys.exit(1)

    model = load_model(args)
    d_in = model.d_in
    num_latents = model.num_latents
    k = model.cfg.k
    sae_type = model.cfg.sae_type.value if hasattr(model.cfg.sae_type, 'value') else str(model.cfg.sae_type)

    print(f"\nModel: d_in={d_in}, num_latents={num_latents}, k={k}, type={sae_type}")
    print(f"Encoder params: {d_in * num_latents:,} ({d_in}x{num_latents})")
    print(f"Samples: {args.n_samples:,}\n")

    # Detect devices
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    if torch.backends.mps.is_available():
        devices.append(torch.device("mps"))
        print("MPS device: Apple Silicon GPU")

    # Benchmark with data transfer (realistic scenario — data comes from numpy memmap)
    print("=" * 80)
    print("WITH DATA TRANSFER (CPU→device per batch, results back to CPU)")
    print("=" * 80)
    print(f"{'Device':<10} {'Batch':<8} {'Time (s)':<10} {'Throughput':>14} {'Samples/s':>12}")
    print("-" * 80)

    best_results = {}

    for device in devices:
        best_tp = 0
        best_bs = 0
        for bs in args.batch_sizes:
            if bs > args.n_samples:
                continue
            try:
                elapsed, throughput = benchmark_encode(model, device, d_in, args.n_samples, bs)
                tag = ""
                if throughput > best_tp:
                    best_tp = throughput
                    best_bs = bs
                    tag = " *"
                print(f"{str(device):<10} {bs:<8} {elapsed:<10.3f} {throughput:>11,.0f}/s{tag:>5}")
            except Exception as e:
                print(f"{str(device):<10} {bs:<8} ERROR: {e}")
        if best_tp > 0:
            best_results[str(device)] = (best_tp, best_bs)
            print(f"  -> Best: {best_tp:,.0f}/s at batch_size={best_bs}")
        print()

    # Benchmark without data transfer (if GPU available)
    gpu_devices = [d for d in devices if d.type != "cpu"]
    if gpu_devices:
        print("=" * 80)
        print("WITHOUT DATA TRANSFER (data pre-loaded on device)")
        print("=" * 80)
        print(f"{'Device':<10} {'Batch':<8} {'Time (s)':<10} {'Throughput':>14}")
        print("-" * 80)

        for device in gpu_devices:
            best_tp = 0
            for bs in args.batch_sizes:
                if bs > args.n_samples:
                    continue
                try:
                    elapsed, throughput = benchmark_encode_no_transfer(model, device, d_in, args.n_samples, bs)
                    tag = ""
                    if throughput > best_tp:
                        best_tp = throughput
                        tag = " *"
                    print(f"{str(device):<10} {bs:<8} {elapsed:<10.3f} {throughput:>11,.0f}/s{tag:>5}")
                except Exception as e:
                    print(f"{str(device):<10} {bs:<8} ERROR: {e}")
            print()

    # Cost analysis
    print("=" * 80)
    print("COST ESTIMATE (per million embeddings)")
    print("=" * 80)

    cost_rates = {
        "cpu": 0.07,     # Modal CPU: ~$0.07/hr per core (8 cores ~ $0.56/hr)
        "cuda": 1.00,    # Modal A10G: ~$1.00/hr
        "mps": 0.00,     # Local Mac: free
    }

    for device_name, (throughput, batch_size) in best_results.items():
        device_type = device_name.split(":")[0]
        rate = cost_rates.get(device_type, 0)
        time_per_million = 1_000_000 / throughput  # seconds
        hours_per_million = time_per_million / 3600
        cost_per_million = hours_per_million * rate

        print(f"  {device_name:<10} {throughput:>10,.0f}/s  "
              f"{time_per_million:>6.1f}s/1M  "
              f"${cost_per_million:.4f}/1M  "
              f"(at ${rate:.2f}/hr)")


if __name__ == "__main__":
    main()
