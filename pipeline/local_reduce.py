"""Local top-10 reduce: download chunk text from Modal and hydrate top-10 indices.

Replaces top10reduce.py when Modal billing is exhausted.
Downloads only the specific shard files needed for hydration.

Usage:
    python -m pipeline.local_reduce \
        --indices pipeline/data/top10_indices_128_8.parquet \
        --output pipeline/data/samples_128_8_combined.parquet
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd


# Shard name → (Modal volume, directory prefix)
def resolve_shard_source(shard_name: str):
    """Map a shard filename to its Modal volume and path."""
    if "of-00099" in shard_name:
        return "embedding-fineweb-edu", "fineweb-edu-sample-10BT-chunked-120/train"
    elif "of-00150" in shard_name:
        return "datasets", "RedPajama-Data-V2-sample-10B-chunked-120/train"
    elif "of-01987" in shard_name:
        return "datasets", "pile-uncopyrighted-chunked-120/train"
    else:
        raise ValueError(f"Unknown shard source: {shard_name}")


def download_shard(volume: str, remote_path: str, local_path: str, modal_bin: str):
    """Download a single shard from Modal volume."""
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if os.path.exists(local_path):
        return  # already cached
    cmd = [modal_bin, "volume", "get", volume, remote_path, local_path]
    subprocess.run(cmd, check=True, capture_output=True)


def download_sae_shard(shard_name: str, sae_slug: str, local_dir: str, modal_bin: str):
    """Download SAE features for a shard (for top_indices/top_acts)."""
    if "of-00099" in shard_name:
        ds = "fineweb-edu-sample-10BT"
    elif "of-00150" in shard_name:
        ds = "RedPajama-Data-V2-sample-10B"
    elif "of-01987" in shard_name:
        ds = "pile-uncopyrighted"
    else:
        return None

    remote = f"{ds}-chunked-120-all-MiniLM-L6-v2-{sae_slug}/train/{shard_name.replace('.parquet', '.parquet')}"
    local = os.path.join(local_dir, "sae", shard_name)
    os.makedirs(os.path.dirname(local), exist_ok=True)
    if not os.path.exists(local):
        try:
            cmd = [modal_bin, "volume", "get", "embeddings", remote, local]
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            return None
    return local


def main():
    parser = argparse.ArgumentParser(description="Local top-10 reduce with Modal downloads")
    parser.add_argument("--indices", required=True, help="Top-10 indices parquet from local merge")
    parser.add_argument("--output", required=True, help="Output samples parquet")
    parser.add_argument("--sae-slug", default="128_8", help="SAE slug for features lookup")
    parser.add_argument("--modal-bin", default=None, help="Path to modal binary")
    parser.add_argument("--cache-dir", default="pipeline/data/shard_cache", help="Cache dir for downloaded shards")
    args = parser.parse_args()

    modal_bin = args.modal_bin
    if not modal_bin:
        # Try to find modal
        for path in ["./venv/bin/modal", os.path.expanduser("~/code/latent-data-modal/venv/bin/modal")]:
            if os.path.exists(path):
                modal_bin = path
                break
        if not modal_bin:
            import shutil
            modal_bin = shutil.which("modal")
    if not modal_bin:
        print("Error: modal CLI not found. Provide --modal-bin")
        return

    print(f"Loading indices from {args.indices}...")
    top10 = pd.read_parquet(args.indices)
    print(f"  {len(top10)} rows, {top10['feature'].nunique()} features")

    shards = top10["shard"].unique()
    print(f"  {len(shards)} shards to hydrate")

    results = []
    for i, shard in enumerate(sorted(shards)):
        rows = top10[top10["shard"] == shard]
        indices = rows["index"].tolist()

        volume, prefix = resolve_shard_source(shard)
        remote_path = f"{prefix}/{shard}"
        local_path = os.path.join(args.cache_dir, "chunks", shard)

        print(f"  [{i+1}/{len(shards)}] {shard}: {len(indices)} rows (volume={volume})")

        try:
            download_shard(volume, remote_path, local_path, modal_bin)
        except subprocess.CalledProcessError as e:
            print(f"    SKIP: download failed ({e})")
            continue

        # Read chunk text
        chunk_df = pd.read_parquet(local_path)
        selected = chunk_df.iloc[indices].copy()
        selected["feature"] = rows["feature"].tolist()
        selected["activation"] = rows["activation"].tolist()

        # Try to get SAE features (top_indices, top_acts) for this shard
        sae_local = download_sae_shard(shard, args.sae_slug, args.cache_dir, modal_bin)
        if sae_local:
            try:
                sae_df = pd.read_parquet(sae_local)
                selected["top_indices"] = [sae_df.iloc[idx]["top_indices"] for idx in indices]
                selected["top_acts"] = [sae_df.iloc[idx]["top_acts"] for idx in indices]
            except Exception:
                pass  # Skip SAE features if unavailable

        results.append(selected)

    if not results:
        print("No results! Check Modal volume access.")
        return

    final = pd.concat(results, ignore_index=True)
    final = final.sort_values(["feature", "activation"], ascending=[True, False])

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    final.to_parquet(args.output, index=False)

    n_features = final["feature"].nunique()
    samples_per = final.groupby("feature").size()
    print(f"\nDone: {len(final)} samples, {n_features} features")
    print(f"Samples per feature: min={samples_per.min()}, max={samples_per.max()}, mean={samples_per.mean():.1f}")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
