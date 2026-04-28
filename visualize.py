#!/usr/bin/env python3
"""
visualize.py — RL run log visualizer
Usage: python visualize.py --dir <run_log_directory>

Expects .jsonl files named like: {strategy}_{seed}_{episodes}.jsonl
where strategy may itself contain underscores (e.g. epsilon_greedy).
Output: runs_viz.html saved in the current working directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Parsing helpers
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
    Returns an ordered list of run dicts:
      { "key": str, "strategy": str, "seed": int|None, "records": [...] }
    Each file is its own run — keyed by the full stem so that files that
    share a strategy+seed but differ in episode count stay separate.
    """
    seen = {}   # stem -> index in runs list
    runs = []
    dir_path = Path(directory)
    jsonl_files = sorted(dir_path.glob("*.jsonl"))

    if not jsonl_files:
        print(f"No .jsonl files found in {directory}", file=sys.stderr)
        sys.exit(1)

    for path in jsonl_files:
        stem = path.stem
        strategy, seed, _ = parse_stem(stem)
        records = load_jsonl(str(path))
        if stem in seen:
            runs[seen[stem]]["records"].extend(records)
        else:
            seen[stem] = len(runs)
            runs.append({
                "key":      stem,
                "strategy": strategy,
                "seed":     seed,
                "records":  records,
            })

    return runs

# Data extraction
METRICS = [
    ("reward",        "episode",     "Reward",           "Episode"),
    ("reward",        "total_steps", "Reward",           "Total Steps"),
    ("ep_len",        "episode",     "Episode Length",   "Episode"),
    ("loss",          "episode",     "Loss",             "Episode"),
    ("mean_loss",     "episode",     "Mean Loss",        "Episode"),
    ("regret",        "episode",     "Regret",           "Episode"),
    ("mean_regret",   "episode",     "Mean Regret",      "Episode"),
    ("entropy",       "episode",     "Entropy",          "Episode"),
    ("mean_entropy",  "episode",     "Mean Entropy",     "Episode"),
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
    charts = []
    for y_key, x_key, y_label, x_label in METRICS:
        traces = []
        for run in runs:
            xs, ys = extract_series(run["records"], y_key, x_key)
            if not xs:
                continue
            strategy, seed, episodes = parse_stem(run["key"])
            display = strategy.replace("_", " ")
            if seed is not None:
                display += f" s={seed}"
            if episodes is not None:
                display += f" ep={episodes}"
            traces.append({
                "x":        xs,
                "y":        ys,
                "run_key":  run["key"],
                "name":     display,
                "hovertemplate": (
                    f"<b>{display}</b><br>"
                    f"{x_label}: %{{x}}<br>"
                    f"{y_label}: %{{y:.4f}}<extra></extra>"
                ),
            })
        if traces:
            charts.append({
                "y_key":   y_key,
                "x_key":   x_key,
                "y_label": y_label,
                "x_label": x_label,
                "traces":  traces,
            })
    return charts

# HTML generation
PALETTE = [
    "#3b82f6", "#f43f5e", "#10b981", "#f59e0b",
    "#a855f7", "#06b6d4", "#fb923c", "#84cc16",
    "#ec4899", "#14b8a6",
]


def generate_html(runs, charts) -> str:
    # One color per unique run (strategy+seed pair)
    color_map = {run["key"]: PALETTE[i % len(PALETTE)] for i, run in enumerate(runs)}

    js_charts = []
    for c in charts:
        js_traces = []
        for t in c["traces"]:
            color = color_map.get(t["run_key"], "#888")
            js_traces.append({
                "x":              t["x"],
                "y":              t["y"],
                "run_key":        t["run_key"],
                "name":           t["name"],
                "hovertemplate":  t["hovertemplate"],
                "color":          color,
            })
        js_charts.append({
            "id":      f"chart_{c['y_key']}_{c['x_key']}",
            "title":   f"{c['y_label']} / {c['x_label']}",
            "x_label": c["x_label"],
            "y_label": c["y_label"],
            "traces":  js_traces,
        })

    # Legend entries — ordered by run, show "strategy seed=N" display label
    legend_entries = []
    for run in runs:
        strategy, seed, episodes = parse_stem(run["key"])
        label = run["strategy"].replace("_", " ")
        if run["seed"] is not None:
            label += f" s={run['seed']}"
        if episodes is not None:
            label += f" ep={episodes}"
        legend_entries.append({
            "key":   run["key"],
            "label": label,
            "color": color_map[run["key"]],
        })

    charts_json  = json.dumps(js_charts)
    legend_json  = json.dumps(legend_entries)
    color_json   = json.dumps(color_map)

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Exploration Visualization</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

:root[data-theme="dark"] {{
  --bg:      #111214;
  --surface: #18191d;
  --border:  #26282e;
  --text:    #e2e4ea;
  --sub:     #6b7080;
  --grid:    #1d1f24;
}}
:root[data-theme="light"] {{
  --bg:      #f5f5f7;
  --surface: #ffffff;
  --border:  #e0e1e6;
  --text:    #1a1b1e;
  --sub:     #8a8f9e;
  --grid:    #eeeff2;
}}

html {{ scroll-behavior: smooth; }}
body {{
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', sans-serif;
  font-size: 13px;
  line-height: 1.5;
  min-height: 100vh;
  transition: background .18s, color .18s;
}}

/* ── Top bar ───────────────────────────────────────────── */
.topbar {{
  padding: 40px 52px 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
}}
.topbar h1 {{
  font-size: 17px;
  font-weight: 500;
  letter-spacing: -.02em;
  color: var(--text);
}}

/* ── Theme toggle ──────────────────────────────────────── */
.toggle {{
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
}}
.toggle-track {{
  width: 34px; height: 18px;
  background: var(--border);
  border-radius: 9px;
  position: relative;
  transition: background .18s;
}}
.toggle-thumb {{
  width: 12px; height: 12px;
  background: var(--sub);
  border-radius: 50%;
  position: absolute;
  top: 3px; left: 3px;
  transition: transform .18s, background .18s;
}}
[data-theme="light"] .toggle-thumb {{
  transform: translateX(16px);
  background: var(--text);
}}
.toggle-lbl {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--sub);
  min-width: 28px;
}}

/* ── Legend ────────────────────────────────────────────── */
.legend {{
  padding: 20px 52px 0;
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
}}
.legend-item {{
  display: flex;
  align-items: center;
  gap: 8px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 11px;
  color: var(--sub);
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 4px 10px;
  cursor: pointer;
  user-select: none;
  transition: border-color .15s, opacity .15s;
}}
.legend-item:hover {{ border-color: var(--sub); }}
.legend-item.active {{
  color: var(--text);
  border-color: var(--sub);
}}
.legend-item.disabled {{
  opacity: 0.35;
  border-color: var(--border);
  color: var(--sub);
}}
.legend-swatch {{
  width: 12px; height: 3px;
  border-radius: 2px;
  flex-shrink: 0;
  transition: opacity .15s;
}}

/* ── Smoothing control ─────────────────────────────────── */
.controls {{
  padding: 16px 52px 0;
  display: flex;
  align-items: center;
  gap: 10px;
}}
.controls label {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--sub);
  letter-spacing: .04em;
}}
.controls input[type=range] {{
  -webkit-appearance: none;
  appearance: none;
  width: 100px; height: 2px;
  background: var(--border);
  border-radius: 1px;
  outline: none;
  cursor: pointer;
}}
.controls input[type=range]::-webkit-slider-thumb {{
  -webkit-appearance: none;
  width: 12px; height: 12px;
  border-radius: 50%;
  background: var(--sub);
  cursor: pointer;
  transition: background .15s;
}}
.controls input[type=range]:hover::-webkit-slider-thumb {{ background: var(--text); }}
.controls .val {{
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  color: var(--text);
  min-width: 24px;
}}

/* ── Chart grid ────────────────────────────────────────── */
.grid {{
  padding: 24px 52px 64px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
  gap: 18px;
}}
.card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  overflow: hidden;
}}
.card-title {{
  padding: 11px 16px;
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px;
  font-weight: 500;
  letter-spacing: .07em;
  text-transform: uppercase;
  color: var(--sub);
  border-bottom: 1px solid var(--border);
}}
.chart-wrap {{ height: 250px; }}
</style>
</head>
<body>

<div class="topbar">
  <h1>Exploration Visualization</h1>
  <div class="toggle" id="themeToggle">
    <div class="toggle-track"><div class="toggle-thumb"></div></div>
    <span class="toggle-lbl" id="toggleLbl">light</span>
  </div>
</div>

<div class="legend" id="legend"></div>

<div class="controls">
  <label for="smoothSlider">window avg</label>
  <input type="range" id="smoothSlider" min="1" max="100" value="20" step="1"/>
  <span class="val" id="smoothVal">20</span>
</div>

<div class="grid" id="charts"></div>

<script>
const CHARTS       = {charts_json};
const LEGEND_ITEMS = {legend_json};
const COLOR_MAP    = {color_json};

// ── Theme ──────────────────────────────────────────────
const root      = document.documentElement;
const toggleBtn = document.getElementById('themeToggle');
const toggleLbl = document.getElementById('toggleLbl');

function setTheme(t) {{
  root.setAttribute('data-theme', t);
  toggleLbl.textContent = t === 'dark' ? 'light' : 'dark';
  replotAll();
}}
toggleBtn.addEventListener('click', () =>
  setTheme(root.getAttribute('data-theme') === 'dark' ? 'light' : 'dark'));

function tv() {{
  const s = getComputedStyle(root);
  const g = k => s.getPropertyValue(k).trim();
  return {{ surface: g('--surface'), grid: g('--grid'), border: g('--border'), sub: g('--sub'), text: g('--text') }};
}}

// ── Window average ──────────────────────────────────────
function wavg(arr, w) {{
  if (w <= 1) return arr;
  const out = new Array(arr.length);
  const half = Math.floor(w / 2);
  for (let i = 0; i < arr.length; i++) {{
    const a = Math.max(0, i - half);
    const b = Math.min(arr.length, i + half + 1);
    let s = 0; for (let j = a; j < b; j++) s += arr[j];
    out[i] = s / (b - a);
  }}
  return out;
}}

let smoothW = 20;
const slider   = document.getElementById('smoothSlider');
const smoothEl = document.getElementById('smoothVal');
slider.addEventListener('input', () => {{
  smoothW = +slider.value;
  smoothEl.textContent = smoothW;
  replotAll();
}});

// ── Legend toggle state ─────────────────────────────────
// All runs visible by default
const hidden = new Set();  // run_key strings that are toggled off

const legendEl = document.getElementById('legend');
LEGEND_ITEMS.forEach(item => {{
  const el = document.createElement('div');
  el.className = 'legend-item active';
  el.dataset.key = item.key;
  el.innerHTML = `<div class="legend-swatch" style="background:${{item.color}}"></div>${{item.label}}`;
  el.addEventListener('click', () => {{
    if (hidden.has(item.key)) {{
      hidden.delete(item.key);
      el.classList.remove('disabled');
      el.classList.add('active');
    }} else {{
      hidden.add(item.key);
      el.classList.remove('active');
      el.classList.add('disabled');
    }}
    replotAll();
  }});
  legendEl.appendChild(el);
}});

// ── Plotly layout ───────────────────────────────────────
function layout(xLabel, yLabel) {{
  const v = tv();
  return {{
    margin: {{ t: 10, r: 14, b: 42, l: 52 }},
    paper_bgcolor: 'transparent',
    plot_bgcolor:  'transparent',
    font: {{ family: "'JetBrains Mono', monospace", size: 9, color: v.sub }},
    xaxis: {{
      title: {{ text: xLabel, standoff: 5, font: {{ size: 9, color: v.sub }} }},
      gridcolor: v.grid, linecolor: v.border, tickcolor: v.border,
      tickfont: {{ size: 8 }}, zeroline: false,
    }},
    yaxis: {{
      title: {{ text: yLabel, standoff: 5, font: {{ size: 9, color: v.sub }} }},
      gridcolor: v.grid, linecolor: v.border, tickcolor: v.border,
      tickfont: {{ size: 8 }}, zeroline: false,
    }},
    legend: {{ visible: false }},
    hovermode: 'x unified',
    hoverlabel: {{
      bgcolor: v.surface, bordercolor: v.border,
      font: {{ family: "'JetBrains Mono', monospace", size: 10 }},
    }},
  }};
}}

const CFG = {{ displayModeBar: false, responsive: true }};

// ── Build traces (respects hidden set) ─────────────────
function makeTraces(c) {{
  return c.traces.map(t => {{
    const isHidden = hidden.has(t.run_key);
    return {{
      x:             t.x,
      y:             wavg(t.y, smoothW),
      name:          t.name,
      type:          'scatter',
      mode:          'lines',
      visible:       isHidden ? false : true,
      line:          {{ color: t.color, width: 1.6 }},
      hovertemplate: t.hovertemplate,
    }};
  }});
}}

// ── Render ──────────────────────────────────────────────
const gridEl = document.getElementById('charts');

CHARTS.forEach(c => {{
  const card = document.createElement('div');
  card.className = 'card';
  card.innerHTML = `<div class="card-title">${{c.title}}</div><div class="chart-wrap" id="${{c.id}}"></div>`;
  gridEl.appendChild(card);
  Plotly.newPlot(c.id, makeTraces(c), layout(c.x_label, c.y_label), CFG);
}});

function replotAll() {{
  CHARTS.forEach(c => Plotly.react(c.id, makeTraces(c), layout(c.x_label, c.y_label), CFG));
}}
</script>
</body>
</html>"""

# Entry point
def main():
    parser = argparse.ArgumentParser(
        description="Visualize RL run logs from a directory of .jsonl files."
    )
    parser.add_argument("--dir", required=True, help="Directory containing .jsonl run logs")
    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        print(f"Error: '{args.dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning {args.dir} …")
    runs = collect_runs(args.dir)
    print(f"  Found {len(runs)} run(s):")
    for r in runs:
        print(f"    {r['key']}  ({len(r['records'])} records)")

    charts = build_chart_data(runs)
    print(f"  Built {len(charts)} chart(s): {[c['y_key'] + ' vs ' + c['x_key'] for c in charts]}")

    html = generate_html(runs, charts)

    out_path = "runs_viz.html"
    with open(out_path, "w") as f:
        f.write(html)

    print(f"  Saved → {os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()