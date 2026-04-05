"""Feature label generation using LLMs.

Generates short descriptive labels for SAE features by showing an LLM
the top-activating text samples, discriminative tokens, and control samples
from similar features.
"""

import asyncio
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline.config import PipelineConfig
from pipeline.llm import LLMClient, LLMConfig, LLMResponse
from pipeline.tokens import compute_all_top_tokens
from pipeline.triage import load_sae_model


# Cost per 1M tokens (USD)
COST_PER_1M_INPUT = {
    "gpt-4.1-mini": 0.40,
    "gpt-4.1-nano": 0.10,
    "claude-haiku-4-5-20251001": 0.80,
    "claude-sonnet-4-5-20250514": 3.00,
}

COST_PER_1M_OUTPUT = {
    "gpt-4.1-mini": 1.60,
    "gpt-4.1-nano": 0.40,
    "claude-haiku-4-5-20251001": 4.00,
    "claude-sonnet-4-5-20250514": 15.00,
}


SYSTEM_PROMPT = """You are analyzing directions in the latent space of a language model. Your task is to identify what concept a direction represents and provide a short label.

You will receive:
1. TOP ACTIVATING SAMPLES: Texts where this direction fires strongly, with discriminative tokens highlighted
2. CONTROL SAMPLES: Texts from similar but distinct directions (these should NOT match the concept)

Analyze what distinguishes the activating samples from the controls. Consider topics, themes, writing styles, domains, or specific subject matter.

EXAMPLE:
Given samples about "Wuthering Heights", "Jane Eyre", "Brontë sisters", with controls about "Victorian architecture", "British monarchy":
LABEL: brontë novels and characters

Now analyze the following direction:"""


def build_user_prompt(
    top_tokens: list[str],
    activating_samples: list[dict],
    control_samples: list[dict],
    max_text_len: int = 300,
) -> str:
    """Build the user prompt for a single feature.

    Args:
        top_tokens: discriminative tokens for this feature
        activating_samples: list of dicts with 'text' and 'activation' keys
        control_samples: list of dicts with 'text' key
        max_text_len: max characters per sample text
    """
    parts = []

    # Top tokens
    parts.append(f"Top discriminative tokens: {', '.join(top_tokens)}")
    parts.append("")

    # Activating samples
    parts.append("Top activating samples:")
    for s in activating_samples:
        text = s["text"][:max_text_len].replace("\n", " ").strip()
        act = s.get("activation", 0)
        parts.append(f'<Sample activation="{act:.3f}">{text}</Sample>')
    parts.append("")

    # Control samples
    if control_samples:
        parts.append("Control samples (similar direction, different concept):")
        for s in control_samples:
            text = s["text"][:max_text_len].replace("\n", " ").strip()
            parts.append(f"<Control>{text}</Control>")
        parts.append("")

    parts.append("Provide a label in 1-6 lowercase words. Respond with only: LABEL: <your label>")

    return "\n".join(parts)


def extract_label(response: str) -> str:
    """Extract label from LLM response.

    Tries to parse 'LABEL: xxx' pattern. Falls back to first line.
    Returns '[FAILED]' for refusal patterns.
    """
    if not response or not response.strip():
        return "[FAILED]"

    # Check for refusal patterns
    refusal_patterns = [
        "please provide",
        "i need more",
        "i cannot",
        "i can't",
        "not enough information",
        "unable to determine",
        "i'm not sure",
        "i don't have enough",
    ]
    response_lower = response.lower().strip()
    for pattern in refusal_patterns:
        if pattern in response_lower:
            return "[FAILED]"

    # Try LABEL: pattern
    match = re.search(r"(?i)LABEL:\s*(.+?)(?:\n|$)", response)
    if match:
        label = match.group(1).strip().strip('"').strip("'").lower()
        if label:
            return label

    # Fallback: first non-empty line
    for line in response.strip().split("\n"):
        line = line.strip().strip('"').strip("'").lower()
        if line and len(line) < 100:
            return line

    return "[FAILED]"


def compute_similar_features(
    W_dec: torch.Tensor, n_neighbors: int = 5, batch_size: int = 256
) -> dict[int, list[int]]:
    """For each feature, find most cosine-similar features by decoder weight.

    Args:
        W_dec: decoder weight matrix [n_features, d_in]
        n_neighbors: number of similar features to return
        batch_size: batch size for cosine similarity computation

    Returns:
        dict mapping feature_id -> list of similar feature_ids
    """
    from sklearn.metrics.pairwise import cosine_similarity

    # Normalize decoder weights
    W = W_dec.detach().cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(W, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    W_norm = W / norms

    n_features = W_norm.shape[0]
    similar = {}

    for start in range(0, n_features, batch_size):
        end = min(start + batch_size, n_features)
        batch = W_norm[start:end]

        # Compute similarities for this batch against all features
        sims = cosine_similarity(batch, W_norm)

        for i, row_idx in enumerate(range(start, end)):
            # Set self-similarity to -inf so we don't pick ourselves
            sims[i, row_idx] = -np.inf
            # Get top-k indices
            top_k = np.argsort(sims[i])[-n_neighbors:][::-1]
            similar[row_idx] = top_k.tolist()

    return similar


def get_control_samples(
    feature_id: int,
    similar_features: dict[int, list[int]],
    samples_df: pd.DataFrame,
    n_per_neighbor: int = 2,
    max_controls: int = 10,
) -> list[dict]:
    """Get control samples from similar features.

    For each similar feature, sample texts that don't also activate the
    current feature.
    """
    neighbors = similar_features.get(feature_id, [])
    controls = []

    for neighbor_id in neighbors:
        neighbor_samples = samples_df[samples_df["feature"] == neighbor_id]
        if neighbor_samples.empty:
            continue

        # Filter out samples that also activate the current feature
        # (check top_indices column if available)
        valid_samples = []
        for _, row in neighbor_samples.iterrows():
            top_indices = row.get("top_indices")
            if top_indices is not None:
                try:
                    if hasattr(top_indices, "__iter__"):
                        indices = list(top_indices)
                    else:
                        indices = []
                    if feature_id in indices:
                        continue
                except (TypeError, ValueError):
                    pass
            valid_samples.append(row)

        if not valid_samples:
            continue

        # Sample n_per_neighbor texts
        rng = np.random.RandomState(feature_id + neighbor_id)
        n_pick = min(n_per_neighbor, len(valid_samples))
        picked = rng.choice(len(valid_samples), size=n_pick, replace=False)
        for idx in picked:
            row = valid_samples[idx]
            text = row.get("chunk_text", "")
            if text:
                controls.append({"text": text})

        if len(controls) >= max_controls:
            break

    return controls[:max_controls]


def estimate_cost(
    model: str,
    total_input_tokens: int,
    total_output_tokens: int,
) -> float:
    """Estimate cost in USD for a given model and token counts."""
    # Check for ollama models (free)
    if model.startswith("ollama/") or "/" not in model and ":" in model:
        return 0.0

    input_cost = COST_PER_1M_INPUT.get(model, 0.0) * total_input_tokens / 1_000_000
    output_cost = COST_PER_1M_OUTPUT.get(model, 0.0) * total_output_tokens / 1_000_000
    return input_cost + output_cost


async def label_features(
    samples_path: str,
    sae_config: PipelineConfig,
    llm_config: LLMConfig,
    output_path: str,
    triage_path: str | None = None,
    num_features: int = 0,
    batch_size: int = 10,
    resume: bool = True,
):
    """Main labeling pipeline.

    Args:
        samples_path: path to top-N samples parquet
        sae_config: config for loading SAE model
        llm_config: config for LLM provider
        output_path: output parquet path
        triage_path: optional triage results parquet (to skip dead features)
        num_features: 0 = all, N = first N (for testing)
        batch_size: concurrent LLM calls
        resume: if True and output exists, skip already-labeled features
    """
    print(f"Loading samples from {samples_path}...")
    samples_df = pd.read_parquet(samples_path)
    print(f"  {len(samples_df)} samples across {samples_df['feature'].nunique()} features")

    # Load SAE model for decoder weights
    print(f"Loading SAE model ({sae_config.model_source}: {sae_config.model_repo}/{sae_config.model_name})...")
    sae_model = load_sae_model(sae_config)
    print(f"  {sae_model.num_latents} features, d_in={sae_model.d_in}")

    # Compute similar features for control samples
    print("Computing feature similarities...")
    similar_features = compute_similar_features(sae_model.W_dec, n_neighbors=5)

    # Compute TF-IDF top tokens for all features
    print("Computing TF-IDF top tokens...")
    all_top_tokens = compute_all_top_tokens(samples_df)
    print(f"  Computed tokens for {len(all_top_tokens)} features")

    # Get feature list
    features = sorted(samples_df["feature"].unique())

    # Filter by triage results if provided
    if triage_path:
        print(f"Loading triage results from {triage_path}...")
        triage_df = pd.read_parquet(triage_path)
        if "status" in triage_df.columns:
            alive = set(
                triage_df[triage_df["status"] != "dead"]["feature"].tolist()
            )
            before = len(features)
            features = [f for f in features if f in alive]
            print(f"  Filtered {before} -> {len(features)} features (skipped dead)")

    # Limit number of features if requested
    if num_features > 0:
        features = features[:num_features]

    print(f"Will label {len(features)} features")

    # Resume: load existing results and skip
    existing_results = []
    already_labeled = set()
    output_file = Path(output_path)
    if resume and output_file.exists():
        existing_df = pd.read_parquet(output_path)
        existing_results = existing_df.to_dict("records")
        already_labeled = set(existing_df["feature"].tolist())
        features = [f for f in features if f not in already_labeled]
        print(f"  Resuming: {len(already_labeled)} already done, {len(features)} remaining")

    if not features:
        print("No features to label. Done.")
        return

    # Initialize LLM client
    llm_client = LLMClient(llm_config)

    # Build prompts for all features
    print("Building prompts...")
    feature_prompts = []
    feature_meta = []
    for feat_id in features:
        feat_samples = samples_df[samples_df["feature"] == feat_id].sort_values(
            "activation", ascending=False
        )
        activating = []
        for _, row in feat_samples.iterrows():
            text = row.get("chunk_text", "")
            act = row.get("activation", 0)
            if text:
                activating.append({"text": text, "activation": act})

        top_tokens = all_top_tokens.get(feat_id, [])
        controls = get_control_samples(feat_id, similar_features, samples_df)

        user_prompt = build_user_prompt(
            top_tokens=top_tokens,
            activating_samples=activating,
            control_samples=controls,
        )

        feature_prompts.append({
            "system": SYSTEM_PROMPT,
            "user": user_prompt,
        })
        feature_meta.append({
            "feature": feat_id,
            "top_tokens": ", ".join(top_tokens),
            "n_activating": len(activating),
            "n_control": len(controls),
        })

    # Process in batches
    all_results = list(existing_results)
    total_batches = (len(features) + batch_size - 1) // batch_size

    print(f"\nLabeling {len(features)} features in {total_batches} batches (batch_size={batch_size})...")
    print(f"Provider: {llm_config.provider}, Model: {llm_config.model}")
    print()

    for batch_idx in range(total_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(features))
        batch_prompts = feature_prompts[start:end]
        batch_meta = feature_meta[start:end]

        responses = await llm_client.complete_batch(
            batch_prompts,
            desc=f"Batch {batch_idx + 1}/{total_batches}",
        )

        for meta, resp in zip(batch_meta, responses):
            label = extract_label(resp.content)
            result = {
                "feature": meta["feature"],
                "label": label,
                "raw_response": resp.content,
                "top_tokens": meta["top_tokens"],
                "n_activating": meta["n_activating"],
                "n_control": meta["n_control"],
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "time_ms": resp.time_ms,
                "provider": llm_config.provider,
                "model": llm_config.model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            all_results.append(result)

        # Save progress every 100 features (less disk I/O)
        if (len(all_results) - len(existing_results)) % 100 < batch_size or end == len(features):
            output_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(all_results).to_parquet(output_path, index=False)

    # Final save
    results_df = pd.DataFrame(all_results)
    results_df.to_parquet(output_path, index=False)

    # Print summary
    new_results = results_df[~results_df["feature"].isin(already_labeled)]
    print_summary(new_results, llm_config.model, llm_config.provider)


def print_summary(results_df: pd.DataFrame, model: str, provider: str):
    """Print a summary of the labeling run."""
    total = len(results_df)
    failed = (results_df["label"] == "[FAILED]").sum()
    success = total - failed

    print("\n" + "=" * 60)
    print("LABELING SUMMARY")
    print("=" * 60)
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Total features labeled: {total}")
    print(f"Successful: {success} ({100 * success / max(total, 1):.1f}%)")
    print(f"Failed: {failed} ({100 * failed / max(total, 1):.1f}%)")

    if total > 0:
        times = results_df["time_ms"]
        print(f"\nTime per feature:")
        print(f"  Mean: {times.mean():.0f}ms")
        print(f"  Median: {times.median():.0f}ms")
        print(f"  Min: {times.min():.0f}ms")
        print(f"  Max: {times.max():.0f}ms")
        print(f"  Total: {times.sum() / 1000:.1f}s")

        total_input = results_df["input_tokens"].sum()
        total_output = results_df["output_tokens"].sum()
        print(f"\nToken usage:")
        print(f"  Total input tokens: {total_input:,}")
        print(f"  Total output tokens: {total_output:,}")

        cost = estimate_cost(model, total_input, total_output)
        if provider == "ollama":
            print(f"\nEstimated cost: $0.00 (local)")
        else:
            print(f"\nEstimated cost: ${cost:.4f}")

    print("=" * 60)


# CLI
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate feature labels using LLMs")
    parser.add_argument("--samples", required=True, help="Path to top-N samples parquet")
    parser.add_argument("--model-source", default="hub", help="SAE model source (hub/local)")
    parser.add_argument("--model-repo", default="enjalot/sae-all-MiniLM-L6-v2-FineWeb-RedPajama-Pile-150M")
    parser.add_argument("--model-name", default="128_4")
    parser.add_argument("--model-path", default=None, help="Local path for SAE model")
    parser.add_argument("--provider", default="ollama", choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--model", default="llama3.1:8b", help="LLM model name")
    parser.add_argument("--output", default="pipeline/output/labels.parquet", help="Output parquet path")
    parser.add_argument("--triage", default=None, help="Triage results parquet (to skip dead features)")
    parser.add_argument("--num-features", type=int, default=0, help="0 = all, N = first N")
    parser.add_argument("--batch-size", type=int, default=None, help="Concurrent LLM calls per batch")
    parser.add_argument("--max-concurrent", type=int, default=None, help="Max concurrent LLM requests")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from existing output")
    args = parser.parse_args()

    # Set default batch size based on provider
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 1 if args.provider == "ollama" else 50

    max_concurrent = args.max_concurrent
    if max_concurrent is None:
        max_concurrent = 1 if args.provider == "ollama" else 20

    sae_config = PipelineConfig(
        model_source=args.model_source,
        model_repo=args.model_repo,
        model_name=args.model_name,
        model_path=args.model_path,
    )

    llm_config = LLMConfig(
        provider=args.provider,
        model=args.model,
        max_concurrent=max_concurrent,
    )

    asyncio.run(
        label_features(
            samples_path=args.samples,
            sae_config=sae_config,
            llm_config=llm_config,
            output_path=args.output,
            triage_path=args.triage,
            num_features=args.num_features,
            batch_size=batch_size,
            resume=not args.no_resume,
        )
    )
