from __future__ import annotations

from pathlib import Path
from typing import Mapping


def write_placeholder_plots(output_dir: str | Path, metrics: Mapping[str, float]) -> Path:
    plots = Path(output_dir) / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    base = float(metrics.get("baseline_total_ttt", 0.0))
    proposed = float(metrics.get("proposed_total_ttt", 0.0))
    max_val = max(base, proposed, 1.0)
    b_h = 160.0 * base / max_val
    p_h = 160.0 * proposed / max_val
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="360" height="240">
<rect width="360" height="240" fill="white"/>
<text x="24" y="28" font-family="Arial" font-size="16">Total TTT Comparison</text>
<rect x="80" y="{200-b_h:.1f}" width="70" height="{b_h:.1f}" fill="#6b7280"/>
<rect x="210" y="{200-p_h:.1f}" width="70" height="{p_h:.1f}" fill="#2563eb"/>
<text x="70" y="222" font-family="Arial" font-size="12">Baseline</text>
<text x="205" y="222" font-family="Arial" font-size="12">Proposed</text>
</svg>
"""
    path = plots / "total_ttt.svg"
    path.write_text(svg, encoding="utf-8")
    return path
