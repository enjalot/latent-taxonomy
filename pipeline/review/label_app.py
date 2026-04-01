"""Label review web app — interactive review of feature labels."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pipeline.label import estimate_cost

# Globals set by CLI
labels_df: pd.DataFrame = None
samples_df: pd.DataFrame = None
similar_features_map: dict = None

app = FastAPI(title="SAE Label Review")

# Mount static files and templates
review_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=review_dir / "static"), name="static")
templates = Jinja2Templates(directory=review_dir / "templates")


def _compute_histogram(values, n_bins: int = 20):
    """Compute histogram data for Chart.js."""
    values = values.dropna()
    if len(values) == 0:
        return {"labels": [], "counts": []}

    min_val = values.min()
    max_val = values.max()
    if min_val == max_val:
        return {"labels": [f"{min_val:.0f}"], "counts": [len(values)]}

    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    counts, _ = np.histogram(values, bins=bin_edges)

    labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(n_bins)]
    return {"labels": labels, "counts": counts.tolist()}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Dashboard with overall stats and charts."""
    df = labels_df
    total = len(df)
    failed = (df["label"] == "[FAILED]").sum()
    success = total - failed
    model = df["model"].iloc[0] if "model" in df.columns and len(df) > 0 else "unknown"
    provider = df["provider"].iloc[0] if "provider" in df.columns and len(df) > 0 else "unknown"

    total_input = int(df["input_tokens"].sum()) if "input_tokens" in df.columns else 0
    total_output = int(df["output_tokens"].sum()) if "output_tokens" in df.columns else 0
    cost = estimate_cost(model, total_input, total_output)
    if provider == "ollama":
        cost = 0.0

    mean_time = df["time_ms"].mean() if "time_ms" in df.columns else 0
    median_time = df["time_ms"].median() if "time_ms" in df.columns else 0

    stats = {
        "provider": provider,
        "model": model,
        "total": total,
        "failed": failed,
        "success_rate": f"{100 * success / max(total, 1):.1f}%",
        "estimated_cost": f"${cost:.4f}",
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "mean_time_ms": f"{mean_time:.0f}",
        "median_time_ms": f"{median_time:.0f}",
    }

    time_data = _compute_histogram(df["time_ms"])

    # Token usage histogram — combine input and output
    token_vals = df[["input_tokens", "output_tokens"]].dropna()
    all_tokens = pd.concat([token_vals["input_tokens"], token_vals["output_tokens"]])
    if len(all_tokens) > 0:
        min_t = all_tokens.min()
        max_t = all_tokens.max()
        n_bins = 20
        bin_edges = np.linspace(min_t, max_t, n_bins + 1)
        input_counts, _ = np.histogram(token_vals["input_tokens"], bins=bin_edges)
        output_counts, _ = np.histogram(token_vals["output_tokens"], bins=bin_edges)
        token_labels = [f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}" for i in range(n_bins)]
        token_data = {
            "labels": token_labels,
            "input_counts": input_counts.tolist(),
            "output_counts": output_counts.tolist(),
        }
    else:
        token_data = {"labels": [], "input_counts": [], "output_counts": []}

    return templates.TemplateResponse("label_dashboard.html", {
        "request": request,
        "stats": stats,
        "time_data": time_data,
        "token_data": token_data,
    })


@app.get("/features", response_class=HTMLResponse)
async def features_list(
    request: Request,
    page: int = Query(1, ge=1),
    status: str = Query("all"),
    search: str = Query(""),
):
    """Paginated feature list with filtering."""
    df = labels_df.copy()
    per_page = 50

    # Filter by status
    if status == "success":
        df = df[df["label"] != "[FAILED]"]
    elif status == "failed":
        df = df[df["label"] == "[FAILED]"]

    # Filter by search
    if search:
        mask = df["label"].str.contains(search, case=False, na=False)
        if "top_tokens" in df.columns:
            mask = mask | df["top_tokens"].str.contains(search, case=False, na=False)
        # Also search by feature ID
        try:
            feat_id = int(search)
            mask = mask | (df["feature"] == feat_id)
        except ValueError:
            pass
        df = df[mask]

    total = len(df)
    total_pages = max(1, math.ceil(total / per_page))
    page = min(page, total_pages)

    start = (page - 1) * per_page
    end = start + per_page
    page_df = df.sort_values("feature").iloc[start:end]

    features = page_df.to_dict("records")

    return templates.TemplateResponse("label_features.html", {
        "request": request,
        "features": features,
        "page": page,
        "total_pages": total_pages,
        "total": total,
        "status": status,
        "search": search,
    })


@app.get("/features/{feature_id}", response_class=HTMLResponse)
async def feature_detail(request: Request, feature_id: int):
    """Detail view for a single feature."""
    label_row = labels_df[labels_df["feature"] == feature_id]
    if label_row.empty:
        return HTMLResponse("<h1>Feature not found</h1>", status_code=404)

    label_info = label_row.iloc[0].to_dict()

    # Get samples for this feature
    feat_samples = []
    if samples_df is not None:
        feat_data = samples_df[samples_df["feature"] == feature_id].sort_values(
            "activation", ascending=False
        )
        feat_samples = feat_data.to_dict("records")

    # Get control samples (from similar features)
    controls = []
    if samples_df is not None and similar_features_map and feature_id in similar_features_map:
        for neighbor_id in similar_features_map[feature_id][:5]:
            neighbor_data = samples_df[samples_df["feature"] == neighbor_id].head(2)
            for _, row in neighbor_data.iterrows():
                controls.append(row.to_dict())

    return templates.TemplateResponse("label_feature_detail.html", {
        "request": request,
        "feature_id": feature_id,
        "label_info": label_info,
        "samples": feat_samples,
        "controls": controls,
    })


def run_app(labels_path: str, samples_path_arg: str | None, port: int = 8503):
    """Load data and start the server."""
    global labels_df, samples_df, similar_features_map

    print(f"Loading labels from {labels_path}...")
    labels_df = pd.read_parquet(labels_path)
    print(f"  {len(labels_df)} labels loaded")

    if samples_path_arg:
        print(f"Loading samples from {samples_path_arg}...")
        samples_df = pd.read_parquet(samples_path_arg)
        print(f"  {len(samples_df)} samples loaded")

    # Try to compute similar features if we have samples
    similar_features_map = {}

    import uvicorn
    print(f"\nStarting label review app at http://localhost:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Label review web app")
    parser.add_argument("--labels", required=True, help="Path to labels parquet")
    parser.add_argument("--samples", default=None, help="Path to samples parquet")
    parser.add_argument("--port", type=int, default=8503, help="Server port")
    args = parser.parse_args()

    run_app(args.labels, args.samples, args.port)
