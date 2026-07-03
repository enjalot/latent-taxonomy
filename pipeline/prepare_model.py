#!/usr/bin/env python
"""
prepare_model.py — build a latent-taxonomy site-data package for a pooled SAE run.

Ports the notebook path (notebooks/umap-top10.ipynb -> umap-grid.ipynb ->
prepare-model.ipynb) to a single resumable CLI. Designed for the exemplar
artifacts produced by latent-sae's experiments.extract_pooled_exemplars
(exemplars_*.parquet + .texts.json + labels_*.parquet).

Stages (each cached in --work-dir; rerun with --stage to redo one):
  embed   : encode the top-N exemplar texts per feature with the source
            encoder (CPU ok), average+normalize per feature (the
            "avg-top-activation" feature embedding), and run the SAE over
            each exemplar embedding to get per-sample top_acts/top_indices.
  umap    : 2D UMAP (x,y == top10_x,top10_y) + 1D UMAP (order), mirroring
            umap-top10.ipynb (n_neighbors=25, min_dist=0.1, cosine, seed 42).
  grid    : linear-assignment alignment of the 2D layout onto a square grid
            (umap-grid.ipynb align_points_to_grid, vectorized).
  package : features.parquet + sharded samples/chunk_N.parquet (100 features
            per shard, ordered by 1-D order) + chunk_mapping.json +
            metadata.json.

Example (MiniLM Stage-J 49K):
  .venv/bin/python pipeline/prepare_model.py \
    --run-dir /data/latent-sae/experiments/results/minilm_l6_stagej_49K_oldrecipe_replay2_k64_20260620_012003 \
    --exemplars exemplars/exemplars_3M.parquet \
    --labels exemplars/labels_full.parquet \
    --name MINILM_STAGEJ_49K \
    --encoder sentence-transformers/all-MiniLM-L6-v2 \
    --work-dir /data/latent-taxonomy/MINILM_STAGEJ_49K/work \
    --out-dir web/public/models/MINILM_STAGEJ_49K \
    --samples-dir /data/latent-taxonomy/MINILM_STAGEJ_49K/samples \
    --meta-json pipeline/model_meta/MINILM_STAGEJ_49K.json \
    --latentsae-path /home/enjalot/code/latent-sae
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# helpers


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_inputs(args):
    """Load exemplars + labels, inner-join on feature, keep top-N chunks."""
    ex_path = Path(args.run_dir) / args.exemplars
    lab_path = Path(args.run_dir) / args.labels
    ex = pd.read_parquet(ex_path, columns=["feature", "max_act", "top_chunks", "top_acts"])
    lab = pd.read_parquet(lab_path, columns=["feature", "label"])
    df = ex.merge(lab, on="feature", how="inner").sort_values("feature").reset_index(drop=True)
    # top_chunks are stored highest-activation first (extract_pooled_exemplars)
    df["top_chunks"] = df["top_chunks"].apply(lambda x: list(x[: args.top_n]))
    df["top_acts"] = df["top_acts"].apply(lambda x: list(x[: args.top_n]))
    log(f"features: {len(ex)} with exemplars, {len(lab)} labeled, {len(df)} joined")
    return df


def load_texts(args, chunk_ids):
    texts_path = Path(args.run_dir) / (args.exemplars + ".texts.json")
    log(f"loading texts map {texts_path} ...")
    with open(texts_path) as f:
        text_map = json.load(f)
    texts = [text_map[str(c)] for c in chunk_ids]
    del text_map
    return texts


# ----------------------------------------------------------------------------
# stage: embed


def stage_embed(args, work):
    import torch

    torch.set_num_threads(max(1, (os.cpu_count() or 8) - 2))
    df = load_inputs(args)
    feats = df["feature"].to_numpy(dtype=np.int64)
    np.save(work / "feature_ids.npy", feats)

    # unique chunk ids across all top-N lists
    all_chunks = np.concatenate([np.asarray(c, dtype=np.int64) for c in df["top_chunks"]])
    uniq = np.unique(all_chunks)
    np.save(work / "chunk_ids.npy", uniq)
    log(f"{len(all_chunks)} exemplar slots -> {len(uniq)} unique chunks to encode")

    texts = load_texts(args, uniq)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.encoder, device="cpu")
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,  # matches extract_pooled_exemplars
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype(np.float32)
    log(f"encoded {len(texts)} texts in {time.time()-t0:.0f}s -> {emb.shape}")
    np.save(work / "chunk_emb.npy", emb)
    del model, texts

    # per-feature averaged (then re-normalized) embedding — umap-top10.ipynb recipe
    pos = {int(c): i for i, c in enumerate(uniq)}
    avg = np.zeros((len(df), emb.shape[1]), dtype=np.float32)
    for i, chunks in enumerate(df["top_chunks"]):
        idx = [pos[int(c)] for c in chunks]
        avg[i] = emb[idx].mean(axis=0)
    avg /= np.linalg.norm(avg, axis=1, keepdims=True)
    np.save(work / "feature_avg_emb.npy", avg)
    log(f"feature averaged embeddings: {avg.shape}")

    # SAE forward over each unique exemplar embedding -> per-sample top-k
    ckpt = find_checkpoint(args)
    if ckpt is None:
        log("no SAE checkpoint found — skipping per-sample top_acts/top_indices")
        return
    sys.path.insert(0, args.latentsae_path)
    from latentsae.sae import Sae

    sae = Sae.load_from_disk(ckpt, device="cpu")
    sae.eval()
    log(f"SAE {ckpt}: d_in={sae.d_in} latents={sae.num_latents}")
    accs, idxs = [], []
    with torch.no_grad():
        for i in range(0, len(emb), 4096):
            out = sae.encode(torch.from_numpy(emb[i : i + 4096]).float())
            accs.append(out.top_acts.numpy().astype(np.float32))
            idxs.append(out.top_indices.numpy().astype(np.int32))
            if (i // 4096) % 20 == 0:
                log(f"  sae encode {i}/{len(emb)}")
    np.savez(work / "chunk_sae_topk.npz", top_acts=np.concatenate(accs), top_indices=np.concatenate(idxs))
    log("saved per-chunk SAE top-k")


def find_checkpoint(args):
    if args.checkpoint:
        p = Path(args.run_dir) / args.checkpoint
        return p if (p / "cfg.json").exists() else None
    ckdir = Path(args.run_dir) / "checkpoints"
    if not ckdir.exists():
        return None
    # prefer a named final export over sae_step_* checkpoints
    cands = sorted(p for p in ckdir.glob("*") if (p / "cfg.json").exists())
    named = [p for p in cands if not p.name.startswith("sae_step_")]
    return (named or cands or [None])[-1]


# ----------------------------------------------------------------------------
# stage: umap


def stage_umap(args, work):
    import umap

    avg = np.load(work / "feature_avg_emb.npy")
    log(f"UMAP 2D on {avg.shape} ...")
    xy = umap.UMAP(
        n_neighbors=25, min_dist=0.1, metric="cosine", random_state=42,
        n_components=2, verbose=True,
    ).fit_transform(avg)
    # scale to [-1, 1] (umap-top10.ipynb)
    mn, mx = xy.min(axis=0), xy.max(axis=0)
    xy = 2 * (xy - mn) / (mx - mn) - 1
    np.save(work / "umap_xy.npy", xy.astype(np.float32))
    log("UMAP 2D done")

    log("UMAP 1D (order) ...")
    order = umap.UMAP(
        n_neighbors=25, min_dist=0.1, metric="cosine", random_state=42,
        n_components=1, verbose=True,
    ).fit_transform(avg)[:, 0]
    np.save(work / "order_1d.npy", order.astype(np.float32))
    log("UMAP 1D done")


# ----------------------------------------------------------------------------
# stage: grid (vectorized port of umap-grid.ipynb align_points_to_grid)


def align_points_to_grid(points, grid_wh, chunk_size=1000):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial import cKDTree

    n = len(points)
    pmin, pmax = points.min(axis=0), points.max(axis=0)
    pnorm = (points - pmin) / (pmax - pmin)

    gx = np.linspace(0, 1, grid_wh)
    gy = np.linspace(0, 1, grid_wh)
    xx, yy = np.meshgrid(gx, gy)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    log(f"grid {grid_wh}x{grid_wh} = {len(grid)} cells for {n} points")

    if len(grid) > n:
        # keep the n grid cells closest to the data (same heuristic as notebook)
        tree = cKDTree(pnorm)
        dist, _ = tree.query(grid, k=1, workers=-1)
        grid = grid[np.argsort(dist)[:n]]
    elif len(grid) < n:
        raise ValueError(f"grid too small: {len(grid)} cells < {n} points")

    # full cost matrix in float32, computed blockwise: n x n
    log("computing cost matrix ...")
    cost = np.empty((n, n), dtype=np.float32)
    for i in range(0, n, 2048):
        d = pnorm[i : i + 2048, None, :] - grid[None, :, :]
        cost[i : i + 2048] = np.sqrt((d ** 2).sum(axis=2), dtype=np.float32)

    log(f"chunked linear assignment (chunk={chunk_size}) ...")
    assigned_col = np.full(n, -1, dtype=np.int64)
    remaining = np.arange(n)
    t0 = time.time()
    for start in range(0, n, chunk_size):
        rows = np.arange(start, min(start + chunk_size, n))
        sub = cost[np.ix_(rows, remaining)]
        r, c = linear_sum_assignment(sub)
        assigned_col[rows[r]] = remaining[c]
        keep = np.ones(len(remaining), dtype=bool)
        keep[c] = False
        remaining = remaining[keep]
        if (start // chunk_size) % 5 == 0:
            log(f"  rows {start}-{rows[-1]} assigned, {len(remaining)} cols left, {time.time()-t0:.0f}s")
    aligned = grid[assigned_col]
    return aligned * (pmax - pmin) + pmin


def stage_grid(args, work):
    xy = np.load(work / "umap_xy.npy").astype(np.float64)
    n = len(xy)
    g = args.grid_size or math.ceil(math.sqrt(n * 1.03))
    aligned = align_points_to_grid(xy, g, chunk_size=args.assign_chunk)
    np.save(work / "grid_xy.npy", aligned)
    log(f"grid alignment done -> {work/'grid_xy.npy'}")


# ----------------------------------------------------------------------------
# stage: package


def stage_package(args, work):
    df = load_inputs(args)
    feats = np.load(work / "feature_ids.npy")
    assert np.array_equal(feats, df["feature"].to_numpy()), "feature set changed since embed stage"

    xy = np.load(work / "umap_xy.npy")
    grid_xy = np.load(work / "grid_xy.npy")
    order = np.load(work / "order_1d.npy").astype(np.float64)
    order = (order - order.min()) / (order.max() - order.min())

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = Path(args.samples_dir) if args.samples_dir else out_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    # ---- features.parquet (exact column contract of NOMIC_FWEDU_25k) ----
    feature_df = pd.DataFrame({
        "feature": df["feature"].astype("int64"),
        "max_activation": df["max_act"].astype("float64"),
        "x": xy[:, 0].astype("float32"),
        "y": xy[:, 1].astype("float32"),
        "top10_x": xy[:, 0].astype("float32"),
        "top10_y": xy[:, 1].astype("float32"),
        "label": df["label"].astype(str).str.replace("FINAL: ", "", regex=False).str.strip(),
        "order": order.astype("float32"),
        "grid_x": grid_xy[:, 0].astype("float64"),
        "grid_y": grid_xy[:, 1].astype("float64"),
    })
    feature_df.to_parquet(out_dir / "features.parquet", index=False)
    log(f"wrote {out_dir/'features.parquet'} ({len(feature_df)} rows)")

    # ---- samples: one row per (feature, exemplar), sharded by 1-D order ----
    uniq = np.load(work / "chunk_ids.npy")
    pos = {int(c): i for i, c in enumerate(uniq)}
    texts = load_texts(args, uniq)
    # materialize the npz arrays once — NpzFile.__getitem__ re-decompresses the
    # whole array on every access, which is O(N^2) if done inside the row loop
    topk = None
    if (work / "chunk_sae_topk.npz").exists():
        with np.load(work / "chunk_sae_topk.npz") as z:
            topk = {"top_acts": z["top_acts"], "top_indices": z["top_indices"]}

    rows = {"id": [], "text": [], "url": [], "feature": [], "activation": [],
            "top_acts": [], "top_indices": []}
    for feature, chunks, acts in zip(df["feature"], df["top_chunks"], df["top_acts"]):
        for c, a in zip(chunks, acts):
            i = pos[int(c)]
            rows["id"].append(str(int(c)))
            rows["text"].append(texts[i])
            rows["url"].append("")
            rows["feature"].append(int(feature))
            rows["activation"].append(float(a))
            if topk is not None:
                # sort each sample's top-k descending by activation for display
                ta, ti = topk["top_acts"][i], topk["top_indices"][i]
                srt = np.argsort(-ta)
                rows["top_acts"].append(ta[srt].astype("float64"))
                rows["top_indices"].append(ti[srt].astype("float64"))
            else:
                rows["top_acts"].append(np.array([], dtype="float64"))
                rows["top_indices"].append(np.array([], dtype="float64"))
    samples_df = pd.DataFrame(rows)
    samples_df["feature"] = samples_df["feature"].astype("int64")
    log(f"samples: {len(samples_df)} rows")

    sorted_feature_df = feature_df.sort_values(by="order")
    by_feature = dict(tuple(samples_df.groupby("feature")))
    chunk_mapping = {}
    n_shards = 0
    for i in range(0, len(sorted_feature_df), 100):
        shard_feats = sorted_feature_df.iloc[i : i + 100]["feature"].tolist()
        shard = pd.concat([by_feature[f] for f in shard_feats if f in by_feature])
        shard.to_parquet(samples_dir / f"chunk_{i // 100}.parquet", index=False)
        for f in shard_feats:
            chunk_mapping[int(f)] = i // 100
        n_shards += 1
    with open(out_dir / "chunk_mapping.json", "w") as f:
        json.dump(chunk_mapping, f)
    # keep a copy next to the shards (handy for the HF dataset upload)
    with open(samples_dir / "chunk_mapping.json", "w") as f:
        json.dump(chunk_mapping, f)
    log(f"wrote {n_shards} sample shards -> {samples_dir}, chunk_mapping.json -> {out_dir}")

    # ---- metadata.json ----
    meta = {
        "id": args.model_id or f"{args.name}",
        "name": args.name,
        "num_features_labeled": int(len(feature_df)),
    }
    ckpt = find_checkpoint(args)
    if ckpt is not None:
        cfg = json.load(open(ckpt / "cfg.json"))
        meta.update({
            "topk": cfg.get("k"),
            "expansion": cfg.get("expansion_factor"),
            "d_in": cfg.get("d_in"),
            "num_latents": cfg.get("num_latents") or (cfg.get("d_in", 0) * cfg.get("expansion_factor", 0)),
        })
        if cfg.get("matryoshka_sizes"):
            meta["matryoshka_levels"] = [int(s) for s in cfg["matryoshka_sizes"].split(",")]
            meta["ks"] = [int(s) for s in cfg["matryoshka_ks"].split(",")]
    if args.meta_json:
        meta.update(json.load(open(args.meta_json)))
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote {out_dir/'metadata.json'}")


# ----------------------------------------------------------------------------

STAGES = ["embed", "umap", "grid", "package"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="SAE experiment run dir")
    ap.add_argument("--exemplars", default="exemplars/exemplars_3M.parquet", help="relative to run-dir")
    ap.add_argument("--labels", default="exemplars/labels_full.parquet", help="relative to run-dir")
    ap.add_argument("--checkpoint", default=None, help="checkpoint dir relative to run-dir (default: auto)")
    ap.add_argument("--name", required=True, help="model name == web/public/models/<NAME>")
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--encoder", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--top-n", type=int, default=10, help="exemplars per feature for layout + samples")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--work-dir", required=True, help="cache dir for intermediate arrays")
    ap.add_argument("--out-dir", required=True, help="site model dir (features.parquet, metadata.json, chunk_mapping.json)")
    ap.add_argument("--samples-dir", default=None, help="where sample shards go (default <out-dir>/samples)")
    ap.add_argument("--meta-json", default=None, help="extra metadata to merge (modality, corpus, evals, samples_base_url, ...)")
    ap.add_argument("--grid-size", type=int, default=0, help="grid side length (default: ceil(sqrt(1.03*n)))")
    ap.add_argument("--assign-chunk", type=int, default=1000)
    ap.add_argument("--latentsae-path", default="/home/enjalot/code/latent-sae")
    ap.add_argument("--stage", default="all", help=f"one of {STAGES} or 'all' or comma list")
    args = ap.parse_args()

    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stages = STAGES if args.stage == "all" else args.stage.split(",")
    for s in stages:
        log(f"===== stage: {s} =====")
        {"embed": stage_embed, "umap": stage_umap, "grid": stage_grid, "package": stage_package}[s](args, work)
    log("done")


if __name__ == "__main__":
    main()
