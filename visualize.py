#!/usr/bin/env python3
"""
visualize.py — RL run log visualizer
Usage: python visualize.py --dir <run_log_directory> [--output <output.html>]

Expects .jsonl files named like: {strategy}_{seed}_{episodes}.jsonl
where strategy may itself contain underscores (e.g. epsilon_greedy).
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_stem(stem: str):
    """
    Extract (strategy, seed, episodes) from a stem like
      epsilon_greedy_42_1000  or  ucb_0_500
    The last two tokens are always seed (int) and episodes (int).
    Everything before them is the strategy name.
    """
    parts = stem.split("_")
    if len(parts) < 3:
        return stem, None, None
    try:
        episodes = int(parts[-1])
        seed = int(parts[-2])
        strategy = "_".join(parts[:-2])
        return strategy, seed, episodes
    except ValueError:
        return stem, None, None


def load_jsonl(path: str):
    records = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def collect_runs(directory: str):
    """
    Returns a dict keyed by (strategy, seed) → list of records,
    plus metadata.
    """
    runs = {}
    dir_path = Path(directory)
    jsonl_files = sorted(dir_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {directory}", file=sys.stderr)
        sys.exit(1)

    for path in jsonl_files:
        stem = path.stem
        strategy, seed, episodes = parse_stem(stem)
        records = load_jsonl(str(path))
        key = (strategy, seed)
        if key in runs:
            runs[key].extend(records)
        else:
            runs[key] = records

    return runs


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

METRICS = [
    ("reward",     "episode",     "Reward",          "Episode"),
    ("reward",     "total_steps", "Reward",          "Total Steps"),
    ("ep_len",     "episode",     "Episode Length",  "Episode"),
    ("loss",       "episode",     "Loss",            "Episode"),
    ("regret",     "episode",     "Regret",          "Episode"),
    ("entropy",    "episode",     "Entropy",         "Episode"),
]


def extract_series(records, y_key, x_key):
    xs, ys = [], []
    for r in records:
        x = r.get(x_key)
        y = r.get(y_key)
        if x is not None and y is not None:
            xs.append(x)
            ys.append(y)
    return xs, ys


def build_chart_data(runs):
    """Build per-metric chart data structures for Plotly."""
    charts = []
    for y_key, x_key, y_label, x_label in METRICS:
        traces = []
        for (strategy, seed), records in runs.items():
            xs, ys = extract_series(records, y_key, x_key)
            if not xs:
                continue
            label = strategy if seed is None else f"{strategy} (seed={seed})"
            traces.append({
                "x": xs,
                "y": ys,
                "name": label,
                "strategy": strategy,
            })
        if traces:
            charts.append({
                "y_key": y_key,
                "x_key": x_key,
                "y_label": y_label,
                "x_label": x_label,
                "traces": traces,
            })
    return charts


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

PALETTE = [
    "#2196f3", "#e91e63", "#4caf50", "#ff9800",
    "#9c27b0", "#00bcd4", "#f44336", "#8bc34a",
    "#ff5722", "#607d8b",
]


def generate_html(runs, charts, source_dir: str) -> str:
    # Assign stable colors per strategy
    strategies = sorted({s for (s, _) in runs.keys()})
    color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(strategies)}

    # Serialise chart data as JS
    js_charts = []
    for c in charts:
        js_traces = []
        for t in c["traces"]:
            color = color_map.get(t["strategy"], "#888")
            js_traces.append({
                "x": t["x"],
                "y": t["y"],
                "name": t["name"],
                "type": "scatter",
                "mode": "lines",
                "line": {"color": color, "width": 1.8},
                "hovertemplate": f"<b>{t['name']}</b><br>{c['x_label']}: %{{x}}<br>{c['y_label']}: %{{y:.4f}}<extra></extra>",
            })
        js_charts.append({
            "id": f"chart_{c['y_key']}_{c['x_key']}",
            "title": f"{c['y_label']} vs {c['x_label']}",
            "x_label": c["x_label"],
            "y_label": c["y_label"],
            "traces": js_traces,
        })

    import json as _json
    charts_json = _json.dumps(js_charts)

    run_count = len(runs)
    strategies_str = ", ".join(strategies) if strategies else "—"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>RL Run Visualizer</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  /* ── Reset & base ─────────────────────────────────────────────────── */
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  :root {{
    --bg:        #0d0f12;
    --surface:   #141720;
    --border:    #222631;
    --text:      #d4d8e2;
    --muted:     #5a6072;
    --accent:    #2196f3;
    --mono:      "JetBrains Mono", "Fira Mono", monospace;
    --sans:      "IBM Plex Sans", "Inter", sans-serif;
  }}

  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500&family=IBM+Plex+Mono:wght@400;500&display=swap');

  html {{ scroll-behavior: smooth; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    line-height: 1.6;
    min-height: 100vh;
  }}

  /* ── Header ───────────────────────────────────────────────────────── */
  header {{
    border-bottom: 1px solid var(--border);
    padding: 32px 48px 28px;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}
  header h1 {{
    font-family: var(--mono);
    font-size: 13px;
    font-weight: 500;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--accent);
  }}
  header p {{
    font-size: 12px;
    color: var(--muted);
    font-family: var(--mono);
  }}
  .meta-pills {{
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin-top: 4px;
  }}
  .pill {{
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 3px;
    padding: 2px 8px;
  }}
  .pill span {{ color: var(--text); }}

  /* ── Legend ───────────────────────────────────────────────────────── */
  .legend-bar {{
    padding: 14px 48px;
    border-bottom: 1px solid var(--border);
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
    align-items: center;
  }}
  .legend-item {{
    display: flex;
    align-items: center;
    gap: 7px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }}
  .legend-dot {{
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }}

  /* ── Grid ─────────────────────────────────────────────────────────── */
  main {{
    padding: 32px 48px 64px;
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(520px, 1fr));
    gap: 24px;
  }}

  .chart-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    transition: border-color .2s;
  }}
  .chart-card:hover {{ border-color: #2d3347; }}

  .chart-header {{
    padding: 14px 18px 10px;
    border-bottom: 1px solid var(--border);
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
    letter-spacing: .06em;
    text-transform: uppercase;
  }}
  .chart-header strong {{
    color: var(--text);
    font-weight: 500;
  }}

  .chart-wrap {{ height: 280px; }}

  /* ── Footer ───────────────────────────────────────────────────────── */
  footer {{
    border-top: 1px solid var(--border);
    padding: 20px 48px;
    font-family: var(--mono);
    font-size: 11px;
    color: var(--muted);
  }}
</style>
</head>
<body>

<header>
  <h1>RL Run Visualizer</h1>
  <p>{source_dir}</p>
  <div class="meta-pills">
    <div class="pill">runs&nbsp;<span>{run_count}</span></div>
    <div class="pill">strategies&nbsp;<span>{strategies_str}</span></div>
  </div>
</header>

<div class="legend-bar" id="legend"></div>

<main id="charts"></main>

<footer>generated by visualize.py</footer>

<script>
const CHARTS = {charts_json};

const PALETTE = {_json.dumps(PALETTE)};
const STRATEGIES = {_json.dumps(strategies)};
const COLOR_MAP = Object.fromEntries(STRATEGIES.map((s,i) => [s, PALETTE[i % PALETTE.length]]));

// Build legend
const legend = document.getElementById('legend');
STRATEGIES.forEach(s => {{
  const el = document.createElement('div');
  el.className = 'legend-item';
  el.innerHTML = `<div class="legend-dot" style="background:${{COLOR_MAP[s]}}"></div>${{s}}`;
  legend.appendChild(el);
}});

// Plotly layout base
function makeLayout(xLabel, yLabel) {{
  return {{
    margin: {{ t: 16, r: 20, b: 44, l: 58 }},
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    font: {{ family: "'IBM Plex Mono', monospace", size: 11, color: '#5a6072' }},
    xaxis: {{
      title: {{ text: xLabel, standoff: 8 }},
      gridcolor: '#1c2030',
      linecolor: '#222631',
      tickcolor: '#222631',
      zeroline: false,
    }},
    yaxis: {{
      title: {{ text: yLabel, standoff: 8 }},
      gridcolor: '#1c2030',
      linecolor: '#222631',
      tickcolor: '#222631',
      zeroline: false,
    }},
    legend: {{ visible: false }},
    hovermode: 'x unified',
    hoverlabel: {{
      bgcolor: '#141720',
      bordercolor: '#2d3347',
      font: {{ family: "'IBM Plex Mono', monospace", size: 11 }},
    }},
  }};
}}

const config = {{
  displayModeBar: true,
  modeBarButtonsToRemove: ['select2d','lasso2d','autoScale2d'],
  displaylogo: false,
  responsive: true,
}};

// Render charts
const main = document.getElementById('charts');
CHARTS.forEach(c => {{
  const card = document.createElement('div');
  card.className = 'chart-card';
  card.innerHTML = `
    <div class="chart-header"><strong>${{c.title}}</strong></div>
    <div class="chart-wrap" id="${{c.id}}"></div>
  `;
  main.appendChild(card);
  Plotly.newPlot(c.id, c.traces, makeLayout(c.x_label, c.y_label), config);
}});
</script>
</body>
</html>"""
    return html


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize RL run logs from a directory of .jsonl files.")
    parser.add_argument("--dir", required=True, help="Directory containing .jsonl run logs")
    parser.add_argument("--output", default="runs_viz.html", help="Output HTML file (default: runs_viz.html)")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: '{args.dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {args.dir} …")
    runs = collect_runs(args.dir)
    print(f"  Found {len(runs)} run(s): {[f'{s}(seed={sd})' for (s,sd) in runs]}")

    charts = build_chart_data(runs)
    print(f"  Built {len(charts)} chart(s): {[c['y_key']+' vs '+c['x_key'] for c in charts]}")

    html = generate_html(runs, charts, os.path.abspath(args.dir))

    out_path = args.output
    with open(out_path, "w") as f:
        f.write(html)

    print(f"  Saved → {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()