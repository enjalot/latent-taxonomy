"""Label comparison tool — generates HTML report comparing multiple label sets."""

import sys
from pathlib import Path

import pandas as pd

from pipeline.label import estimate_cost


def generate_comparison_html(
    label_files: list[str],
    output_path: str,
):
    """Generate a self-contained HTML comparison of multiple label sets.

    Args:
        label_files: list of paths to label parquet files
        output_path: path for output HTML file
    """
    datasets = []
    for path in label_files:
        df = pd.read_parquet(path)
        name = Path(path).stem
        datasets.append({"name": name, "path": path, "df": df})

    # Build summary table data
    summaries = []
    for ds in datasets:
        df = ds["df"]
        total = len(df)
        failed = (df["label"] == "[FAILED]").sum()
        model = df["model"].iloc[0] if "model" in df.columns else "unknown"
        provider = df["provider"].iloc[0] if "provider" in df.columns else "unknown"
        mean_time = df["time_ms"].mean() if "time_ms" in df.columns else 0
        total_input = df["input_tokens"].sum() if "input_tokens" in df.columns else 0
        total_output = df["output_tokens"].sum() if "output_tokens" in df.columns else 0
        cost = estimate_cost(model, int(total_input), int(total_output))
        if provider == "ollama":
            cost = 0.0

        summaries.append({
            "name": ds["name"],
            "provider": provider,
            "model": model,
            "total": total,
            "failed": failed,
            "success_rate": f"{100 * (total - failed) / max(total, 1):.1f}%",
            "mean_time_ms": f"{mean_time:.0f}",
            "total_input_tokens": f"{int(total_input):,}",
            "total_output_tokens": f"{int(total_output):,}",
            "estimated_cost": f"${cost:.4f}",
        })

    # Build feature comparison data
    # Merge all datasets on feature ID
    all_features = set()
    for ds in datasets:
        all_features.update(ds["df"]["feature"].tolist())
    all_features = sorted(all_features)

    # Get top_tokens from first dataset that has them
    tokens_map = {}
    for ds in datasets:
        for _, row in ds["df"].iterrows():
            fid = row["feature"]
            if fid not in tokens_map and row.get("top_tokens"):
                tokens_map[fid] = row["top_tokens"]

    comparison_rows = []
    for feat_id in all_features:
        row = {"feature": feat_id, "top_tokens": tokens_map.get(feat_id, "")}
        for ds in datasets:
            feat_row = ds["df"][ds["df"]["feature"] == feat_id]
            if not feat_row.empty:
                r = feat_row.iloc[0]
                row[f"label_{ds['name']}"] = r.get("label", "")
                row[f"time_{ds['name']}"] = f"{r.get('time_ms', 0):.0f}"
            else:
                row[f"label_{ds['name']}"] = "(missing)"
                row[f"time_{ds['name']}"] = "-"
        comparison_rows.append(row)

    # Generate HTML
    ds_names = [ds["name"] for ds in datasets]
    html = _render_html(summaries, comparison_rows, ds_names)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    print(f"Comparison report saved to {output_path}")


def _render_html(summaries, comparison_rows, ds_names):
    """Render the full HTML comparison page."""
    # Summary table rows
    summary_rows_html = ""
    for s in summaries:
        summary_rows_html += f"""<tr>
            <td>{s['name']}</td>
            <td>{s['provider']}</td>
            <td>{s['model']}</td>
            <td>{s['total']}</td>
            <td>{s['failed']}</td>
            <td>{s['success_rate']}</td>
            <td>{s['mean_time_ms']}ms</td>
            <td>{s['total_input_tokens']}</td>
            <td>{s['total_output_tokens']}</td>
            <td>{s['estimated_cost']}</td>
        </tr>"""

    # Comparison table headers
    comp_headers = "<th>Feature</th><th>Top Tokens</th>"
    for name in ds_names:
        comp_headers += f"<th>Label ({name})</th><th>Time ({name})</th>"

    # Comparison table rows
    comp_rows_html = ""
    for row in comparison_rows:
        cells = f"<td>{row['feature']}</td><td class='tokens'>{row['top_tokens']}</td>"
        for name in ds_names:
            label = row.get(f"label_{name}", "")
            time_val = row.get(f"time_{name}", "")
            label_class = "failed" if label == "[FAILED]" else ""
            cells += f"<td class='{label_class}'>{label}</td><td>{time_val}ms</td>"
        comp_rows_html += f"<tr>{cells}</tr>"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Label Comparison Report</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; padding: 20px; background: #f5f5f5; color: #333; }}
    h1 {{ margin-bottom: 20px; }}
    h2 {{ margin: 30px 0 15px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
    th {{ background: #2c3e50; color: white; padding: 10px 12px; text-align: left; font-size: 13px; cursor: pointer; }}
    th:hover {{ background: #34495e; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #eee; font-size: 13px; }}
    tr:hover {{ background: #f0f7ff; }}
    .failed {{ color: #e74c3c; font-weight: bold; }}
    .tokens {{ color: #7f8c8d; font-size: 11px; max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
    .controls {{ margin: 15px 0; display: flex; gap: 10px; align-items: center; }}
    .controls input {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; width: 300px; }}
    .controls select {{ padding: 8px 12px; border: 1px solid #ddd; border-radius: 4px; font-size: 14px; }}
</style>
</head>
<body>
<h1>Label Comparison Report</h1>

<h2>Summary</h2>
<table id="summary-table">
<thead><tr>
    <th>Name</th><th>Provider</th><th>Model</th><th>Total</th><th>Failed</th>
    <th>Success Rate</th><th>Mean Time</th><th>Input Tokens</th><th>Output Tokens</th><th>Est. Cost</th>
</tr></thead>
<tbody>{summary_rows_html}</tbody>
</table>

<h2>Feature-by-Feature Comparison</h2>
<div class="controls">
    <input type="text" id="search" placeholder="Search by feature ID or label..." oninput="filterTable()">
</div>
<table id="comparison-table">
<thead><tr>{comp_headers}</tr></thead>
<tbody>{comp_rows_html}</tbody>
</table>

<script>
function filterTable() {{
    const query = document.getElementById('search').value.toLowerCase();
    const rows = document.querySelectorAll('#comparison-table tbody tr');
    rows.forEach(row => {{
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(query) ? '' : 'none';
    }});
}}

// Sortable headers
document.querySelectorAll('#comparison-table th').forEach((th, idx) => {{
    th.addEventListener('click', () => {{
        const table = document.getElementById('comparison-table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const dir = th.dataset.dir === 'asc' ? 'desc' : 'asc';
        th.dataset.dir = dir;
        rows.sort((a, b) => {{
            let va = a.cells[idx].textContent.trim();
            let vb = b.cells[idx].textContent.trim();
            const na = parseFloat(va), nb = parseFloat(vb);
            if (!isNaN(na) && !isNaN(nb)) {{
                return dir === 'asc' ? na - nb : nb - na;
            }}
            return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
        }});
        rows.forEach(r => tbody.appendChild(r));
    }});
}});
</script>
</body>
</html>"""


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare multiple label sets")
    parser.add_argument("files", nargs="+", help="Label parquet files to compare")
    parser.add_argument("--output", default="pipeline/output/comparison.html", help="Output HTML path")
    args = parser.parse_args()

    generate_comparison_html(args.files, args.output)
