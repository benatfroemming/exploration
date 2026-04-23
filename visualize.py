#!/usr/bin/env python3
"""
visualize.py  --  DQN training log visualizer

Usage:
    python visualize.py --log path/to/run.jsonl
    python visualize.py --log path/to/run.jsonl --out dashboard.html
    python visualize.py --log path/to/run.jsonl --no-open   # don't auto-open browser
"""

from __future__ import annotations

import argparse
import json
import math
import os
import webbrowser
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────

def load_log(path: str) -> tuple[list[dict], dict | None]:
    """Return (episode_records, summary_record | None)."""
    episodes, summary = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("episode") == "summary":
                summary = rec
            else:
                episodes.append(rec)
    return episodes, summary


def rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    for i, v in enumerate(values):
        start = max(0, i - window + 1)
        out.append(sum(values[start : i + 1]) / (i - start + 1))
    return out


def safe(values: list, key: str) -> list[float]:
    """Extract key from list of dicts, replacing None with NaN for JS."""
    return [v.get(key) for v in values]


def to_js_array(values: list) -> str:
    def fmt(v):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "null"
        return str(v)
    return "[" + ",".join(fmt(v) for v in values) + "]"


# ── HTML generation ───────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>DQN Training Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

  :root {
    --bg:       #0d0f14;
    --surface:  #141720;
    --border:   #252a38;
    --text:     #c9d1e0;
    --muted:    #515a72;
    --accent1:  #4af0a8;   /* reward  */
    --accent2:  #5b9cf6;   /* epsilon */
    --accent3:  #f6a45b;   /* td      */
    --accent4:  #e05bf6;   /* q_diff  */
    --accent5:  #f6f05b;   /* ep_len  */
  }

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'IBM Plex Sans', sans-serif;
    font-weight: 300;
    min-height: 100vh;
    padding: 2rem 2.5rem 4rem;
  }

  header {
    display: flex;
    align-items: baseline;
    gap: 1.5rem;
    margin-bottom: 2.5rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.25rem;
  }
  header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.1rem;
    font-weight: 600;
    letter-spacing: .08em;
    color: var(--accent1);
  }
  header span {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .75rem;
    color: var(--muted);
  }

  .stats-row {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 1rem;
    margin-bottom: 2.5rem;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1rem 1.25rem;
  }
  .stat-card .label {
    font-size: .65rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .4rem;
    font-family: 'IBM Plex Mono', monospace;
  }
  .stat-card .value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.45rem;
    font-weight: 600;
    color: var(--text);
  }
  .stat-card .value.accent1 { color: var(--accent1); }
  .stat-card .value.accent2 { color: var(--accent2); }
  .stat-card .value.accent3 { color: var(--accent3); }
  .stat-card .value.accent4 { color: var(--accent4); }

  .charts-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }
  @media (max-width: 900px) {
    .charts-grid { grid-template-columns: 1fr; }
  }

  .chart-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1.25rem 1.5rem 1.5rem;
  }
  .chart-card.wide {
    grid-column: 1 / -1;
  }
  .chart-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: .7rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1rem;
  }
  .chart-wrap {
    position: relative;
    height: 220px;
  }
  .chart-wrap.tall {
    height: 260px;
  }
</style>
</head>
<body>

<header>
  <h1>DQN · TRAINING LOG</h1>
  <span id="logpath">__LOG_PATH__</span>
</header>

<div class="stats-row" id="statsRow"></div>

<div class="charts-grid">
  <div class="chart-card wide">
    <div class="chart-title">Episode Reward</div>
    <div class="chart-wrap tall"><canvas id="cReward"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">|TD Error|</div>
    <div class="chart-wrap"><canvas id="cTD"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Q<sub>max</sub> − Q<sub>taken</sub></div>
    <div class="chart-wrap"><canvas id="cQDiff"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Epsilon (ε)</div>
    <div class="chart-wrap"><canvas id="cEpsilon"></canvas></div>
  </div>
  <div class="chart-card">
    <div class="chart-title">Episode Length</div>
    <div class="chart-wrap"><canvas id="cEpLen"></canvas></div>
  </div>
</div>

<script>
// ── data injected by Python ──────────────────────────────────────────────────
const episodes  = __EPISODES__;
const rewards   = __REWARDS__;
const td        = __TD__;
const qdiff     = __QDIFF__;
const epsilon   = __EPSILON__;
const eplen     = __EPLEN__;
const steps     = __STEPS__;
const rollReward = __ROLL_REWARD__;
const WINDOW    = __WINDOW__;

// ── summary stats ────────────────────────────────────────────────────────────
const validRewards  = rewards.filter(v => v !== null);
const validTD       = td.filter(v => v !== null);
const validQDiff    = qdiff.filter(v => v !== null);

function fmt(v, dec=3) {
  if (v === null || v === undefined) return "—";
  return typeof v === 'number' ? v.toFixed(dec) : v;
}

const statsRow = document.getElementById("statsRow");
const stats = [
  { label: "Episodes",      value: episodes.length,                      cls: ""        },
  { label: "Total Steps",   value: (steps[steps.length-1]||0).toLocaleString(), cls: "" },
  { label: "Mean Reward",   value: fmt(validRewards.reduce((a,b)=>a+b,0)/validRewards.length, 2), cls: "accent1" },
  { label: "Max Reward",    value: fmt(Math.max(...validRewards), 2),     cls: "accent1" },
  { label: "Mean |TD|",     value: fmt(validTD.reduce((a,b)=>a+b,0)/Math.max(validTD.length,1), 4), cls: "accent3" },
  { label: "Mean Q diff",   value: fmt(validQDiff.reduce((a,b)=>a+b,0)/Math.max(validQDiff.length,1), 4), cls: "accent4" },
  { label: "Final ε",       value: fmt(epsilon[epsilon.length-1], 4),    cls: "accent2" },
];
stats.forEach(s => {
  statsRow.innerHTML += `
    <div class="stat-card">
      <div class="label">${s.label}</div>
      <div class="value ${s.cls}">${s.value}</div>
    </div>`;
});

// ── chart defaults ───────────────────────────────────────────────────────────
Chart.defaults.color = "#515a72";
Chart.defaults.font.family = "'IBM Plex Mono', monospace";
Chart.defaults.font.size = 10;

const gridColor = "#1e2333";
const tickColor = "#515a72";

function baseOpts(yLabel="") {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    plugins: {
      legend: { display: true, labels: { boxWidth: 10, padding: 12 } },
      tooltip: { mode: "index", intersect: false },
    },
    scales: {
      x: {
        ticks: { color: tickColor, maxTicksLimit: 8 },
        grid:  { color: gridColor },
        title: { display: false },
      },
      y: {
        ticks: { color: tickColor, maxTicksLimit: 6 },
        grid:  { color: gridColor },
        title: { display: !!yLabel, text: yLabel, color: tickColor },
      },
    },
  };
}

function makeLine(id, datasets, opts={}) {
  return new Chart(document.getElementById(id), {
    type: "line",
    data: { labels: episodes, datasets },
    options: { ...baseOpts(), ...opts },
  });
}

function dataset(label, data, color, opts={}) {
  return {
    label,
    data,
    borderColor: color,
    backgroundColor: color + "18",
    borderWidth: 1.5,
    pointRadius: 0,
    tension: 0.3,
    fill: false,
    spanGaps: true,
    ...opts,
  };
}

// ── reward ───────────────────────────────────────────────────────────────────
makeLine("cReward", [
  dataset("raw",  rewards,   "var(--accent1)", { borderWidth: 1, opacity: 0.4 }),
  dataset(`roll-${WINDOW}`, rollReward, "var(--accent1)", { borderWidth: 2.5 }),
]);

// ── td ───────────────────────────────────────────────────────────────────────
makeLine("cTD", [
  dataset("|TD|", td, "var(--accent3)"),
]);

// ── q_diff ───────────────────────────────────────────────────────────────────
makeLine("cQDiff", [
  dataset("Qmax−Qtaken", qdiff, "var(--accent4)"),
]);

// ── epsilon ──────────────────────────────────────────────────────────────────
makeLine("cEpsilon", [
  dataset("ε", epsilon, "var(--accent2)"),
]);

// ── ep_len ───────────────────────────────────────────────────────────────────
makeLine("cEpLen", [
  dataset("ep_len", eplen, "var(--accent5)"),
]);
</script>
</body>
</html>
"""


def build_html(episodes: list[dict], summary: dict | None, log_path: str, window: int = 50) -> str:
    eps_idx  = [r["episode"] for r in episodes]
    rewards  = safe(episodes, "reward")
    td       = safe(episodes, "td")
    qdiff    = safe(episodes, "q_diff")
    epsilon  = safe(episodes, "epsilon")
    eplen    = safe(episodes, "ep_len")
    steps    = safe(episodes, "total_steps")

    # rolling mean over rewards (skip None)
    reward_vals = [r if r is not None else float("nan") for r in rewards]
    roll = rolling_mean(reward_vals, window)

    html = HTML_TEMPLATE
    html = html.replace("__LOG_PATH__", log_path)
    html = html.replace("__EPISODES__",   to_js_array(eps_idx))
    html = html.replace("__REWARDS__",    to_js_array(rewards))
    html = html.replace("__TD__",         to_js_array(td))
    html = html.replace("__QDIFF__",      to_js_array(qdiff))
    html = html.replace("__EPSILON__",    to_js_array(epsilon))
    html = html.replace("__EPLEN__",      to_js_array(eplen))
    html = html.replace("__STEPS__",      to_js_array(steps))
    html = html.replace("__ROLL_REWARD__",to_js_array(roll))
    html = html.replace("__WINDOW__",     str(window))
    return html


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize a DQN JSONL training log.")
    parser.add_argument("--log",    required=True, help="Path to the .jsonl log file")
    parser.add_argument("--out",    default=None,  help="Output HTML path (default: <log>.html)")
    parser.add_argument("--window", type=int, default=50, help="Rolling average window (default: 50)")
    parser.add_argument("--no-open", action="store_true", help="Don't auto-open the browser")
    args = parser.parse_args()

    log_path = args.log
    out_path = args.out or str(Path(log_path).with_suffix(".html"))

    print(f"Loading {log_path} …")
    episodes, summary = load_log(log_path)
    print(f"  {len(episodes)} episode records loaded")
    if summary:
        print(f"  Summary record found: {summary}")

    html = build_html(episodes, summary, log_path, window=args.window)

    with open(out_path, "w") as f:
        f.write(html)
    print(f"Dashboard → {out_path}")

    if not args.no_open:
        webbrowser.open(f"file://{os.path.abspath(out_path)}")


if __name__ == "__main__":
    main()