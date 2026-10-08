#!/usr/bin/env python3
"""Polish existing VRAVEN HTML reports in vraven_outputs/ for label clarity.

- Patches Decision Proof + Explanation report CSS (no overlapping headings)
- Injects style so Plotly in-chart titles sit below HTML section headings
- Rebuilds per-model gallery INDEX pages
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vraven_outputs"
sys.path.insert(0, str(ROOT / "scripts"))

DECISION_STYLE = """
:root{
  --canvas:#030A12;--panel:#07111F;--raised:#0B1728;--border:#1C344D;
  --text:#F1F5F9;--muted:#94A3B8;--ink:#F1F5F9;
  --accent:#00D4FF;--accent-grad:linear-gradient(135deg, #00D4FF 0%, #8B5CF6 50%, #FF2D55 100%);
  --good:#32D74B;--warn:#FF8A00;--bad:#FF2D55;
  --shadow:0 14px 40px -10px rgba(0,0,0,.6), 0 0 20px rgba(0,212,255,.04);
}
html[data-theme='light']{
  --canvas:#FFFFFF;--panel:#FFFFFF;--raised:#F1F5F9;--border:#CBD5E1;
  --text:#111827;--muted:#475569;--ink:#111827;
  --accent:#0E7490;--accent-grad:linear-gradient(135deg, #0057B8 0%, #6D28D9 50%, #B42318 100%);
  --good:#15803D;--warn:#C2410C;--bad:#B42318;
  --shadow:0 10px 25px -5px rgba(0,0,0,.05);
}
*{box-sizing:border-box}
body{
  margin:0;background:var(--canvas);
  background-image:radial-gradient(ellipse 80% 50% at 50% -20%, rgba(22, 139, 255, 0.12), transparent 70%), radial-gradient(ellipse 60% 40% at 80% 100%, rgba(139, 92, 246, 0.08), transparent 60%);
  background-attachment:fixed;
  color:var(--text);font-family:"Inter",sans-serif;line-height:1.6;-webkit-font-smoothing:antialiased;
}
main{max-width:1480px;margin:auto;padding:36px 32px 64px}
header,.proof-section,.card{
  background:var(--panel);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
  border:1px solid var(--border);box-shadow:var(--shadow);border-radius:16px;
  transition:border-color 0.2s ease, box-shadow 0.2s ease;
}
header{position:relative;overflow:hidden;padding:30px 32px 26px;margin-bottom:26px}
header::before{content:"";position:absolute;top:0;left:0;right:0;height:4px;background:var(--accent-grad)}
.header-row{display:flex;justify-content:space-between;gap:24px;align-items:flex-start;flex-wrap:wrap}
h1{font-family:"Plus Jakarta Sans",sans-serif;font-size:clamp(24px,3.2vw,32px);margin:10px 0 8px;letter-spacing:-0.02em;line-height:1.2;background:var(--accent-grad);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
h2{font-family:"Plus Jakarta Sans",sans-serif;font-size:14px;text-transform:uppercase;letter-spacing:.08em;margin:0 0 16px;padding-bottom:10px;border-bottom:1px solid var(--border);line-height:1.35;color:var(--accent)}
p{color:var(--muted);overflow-wrap:anywhere;max-width:78ch;font-size:14px}
.tag{display:inline-block;padding:5px 12px;border:1px solid var(--border);border-radius:999px;font-size:12px;font-weight:600;font-family:"Plus Jakarta Sans",sans-serif;margin:0 6px 6px 0;color:var(--muted);background:var(--raised)}
.tag.good{color:var(--good);background:rgba(16,185,129,0.12);border-color:rgba(16,185,129,0.3)}
.tag.warn{color:var(--warn);background:rgba(245,158,11,0.12);border-color:rgba(245,158,11,0.3)}
.shortcut-alert-box{
  background: rgba(244, 63, 94, 0.12);
  border: 1px solid rgba(244, 63, 94, 0.4);
  border-radius: 12px;
  padding: 16px 20px;
  margin: 20px 0;
  display: flex;
  flex-direction: column;
  gap: 6px;
  box-shadow: 0 0 20px rgba(244, 63, 94, 0.15);
}
.shortcut-alert-hdr{
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 0.85rem;
  font-weight: 800;
  color: #F43F5E;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  display: flex;
  align-items: center;
  gap: 8px;
}
.shortcut-alert-body{
  font-size: 0.88rem;
  color: var(--text);
  line-height: 1.5;
}
.ood-outlier-chip{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 8px;
  background: rgba(168, 85, 247, 0.14);
  border: 1px solid rgba(168, 85, 247, 0.35);
  color: #C084FC;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.74rem;
  font-weight: 700;
}
.toggle{display:flex;gap:4px;padding:4px;border:1px solid var(--border);border-radius:12px;background:var(--raised);flex-shrink:0}
button{border:0;background:transparent;color:var(--muted);padding:8px 16px;border-radius:999px;font-family:"Plus Jakarta Sans",sans-serif;font-size:12px;font-weight:700;cursor:pointer;transition:all 180ms ease}
button.active{background:var(--accent-grad);color:#ffffff;box-shadow:0 4px 14px rgba(56,189,248,0.35)}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-bottom:26px}
.card{border-radius:16px;padding:22px 20px 18px;min-width:0;position:relative;overflow:hidden}
.card::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:var(--accent-grad)}
.metric{font-family:"JetBrains Mono",monospace;font-size:clamp(20px,2.4vw,28px);font-weight:750;line-height:1.25;overflow-wrap:anywhere;color:var(--ink)}
.label{font-family:"Plus Jakarta Sans",sans-serif;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;line-height:1.4;margin-bottom:10px;font-weight:700}
.narrative{font-size:15px;color:var(--ink);line-height:1.65}
.proof-section{border-radius:16px;margin:28px 0;padding:24px 22px 18px;overflow:visible}
.section-kicker{font-family:"Plus Jakarta Sans",sans-serif;font-size:11px;text-transform:uppercase;letter-spacing:.12em;font-weight:800;padding:0;margin:0 0 6px;color:var(--accent);line-height:1.4}
.section-title{font-family:"Plus Jakarta Sans",sans-serif;font-size:20px;font-weight:800;letter-spacing:-.01em;margin:0 0 16px;padding:0 0 10px;border-bottom:1px solid var(--border);line-height:1.3;color:var(--ink)}
.viz{width:100%;overflow:visible;min-height:420px;margin-top:8px;border-radius:12px}
.viz .js-plotly-plot,.viz .plotly-graph-div{width:100% !important;max-width:100%;border-radius:12px}
html[data-theme='dark'] .mode-light{display:none}
html[data-theme='light'] .mode-dark{display:none}
table{width:100%;border-collapse:separate;border-spacing:0}
td,th{padding:12px 14px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top;overflow-wrap:anywhere;font-size:13px}
th{color:var(--muted);font-weight:600;font-size:11px;letter-spacing:0.06em;text-transform:uppercase}
td{font-family:"JetBrains Mono",monospace}
.tier{font-family:"JetBrains Mono",monospace;font-size:11px;padding:4px 9px;border-radius:8px;background:var(--raised);border:1px solid var(--border);white-space:nowrap;color:var(--ink)}
pre{white-space:pre-wrap;background:var(--raised);border:1px solid var(--border);padding:18px;border-radius:14px;color:var(--ink);font-family:"JetBrains Mono",monospace;font-size:12px;line-height:1.55;overflow:auto}
details{margin-top:18px}
summary{cursor:pointer;color:var(--ink);font-weight:700;font-family:"Plus Jakarta Sans",sans-serif}
.warning-list{color:var(--warn)}
footer{color:var(--muted);font-size:12px;padding:24px 4px 48px;line-height:1.5;font-family:"Inter",sans-serif}
@media(max-width:980px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:620px){main{padding:14px}.grid{grid-template-columns:1fr}.toggle{width:100%;justify-content:stretch} .toggle button{flex:1}}
"""

PROOF_EXTRA_CSS = """
/* Polished report chrome: ultra-modern design language */
section h2 {
  margin: 0 0 18px 0 !important;
  padding: 0 0 12px 0 !important;
  border-bottom: 1px solid var(--vraven-border) !important;
  font-family: "Plus Jakarta Sans", sans-serif !important;
  font-size: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  line-height: 1.35 !important;
  color: var(--vraven-accent, #00D4FF) !important;
}
.theme-toggle button[aria-pressed="true"] {
  background: var(--vraven-accent-gradient, linear-gradient(135deg, #00D4FF 0%, #8B5CF6 100%)) !important;
  color: #FFFFFF !important;
  box-shadow: 0 4px 12px rgba(0, 212, 255, 0.35) !important;
}
.viz-panel, .viz-panel .js-plotly-plot {
  margin-top: 8px;
  border-radius: 12px;
}
.vraven-evidence-filters {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  align-items: center;
  margin: 0 0 8px;
}
.vraven-filter-chip {
  appearance: none;
  border: 1px solid var(--border, #1C344D);
  background: transparent;
  color: inherit;
  border-radius: 6px;
  padding: 4px 8px;
  font-size: 11px;
  cursor: pointer;
  display: inline-flex;
  gap: 6px;
  align-items: center;
}
.vraven-filter-chip.is-active {
  border-color: #22D3EE;
  color: #22D3EE;
}
.vraven-filter-count {
  margin-left: 4px;
  font-size: 11px;
  opacity: 0.75;
}
.vraven-edge-inspector {
  flex: 0 0 340px;
  max-height: 720px;
  overflow: auto;
  border: 1px solid var(--border, #1C344D);
  border-radius: 8px;
  padding: 12px 14px;
  font-size: 12px;
  line-height: 1.4;
}
.vraven-edge-inspector h3 {
  margin: 0 0 4px;
  font-size: 13px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  opacity: 0.75;
}
.vraven-edge-inspector .hdr {
  font-size: 15px;
  font-weight: 700;
  margin: 0 0 10px;
}
.vraven-edge-inspector .sec {
  margin: 12px 0 0;
  padding-top: 10px;
  border-top: 1px solid rgba(148,163,184,0.25);
}
.vraven-edge-inspector .sec strong {
  display: block;
  margin-bottom: 6px;
  font-size: 11px;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.75;
}
.vraven-edge-inspector .row {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  margin: 3px 0;
}
.vraven-edge-inspector .status {
  margin-top: 12px;
  padding: 8px 10px;
  border-radius: 6px;
  border: 1px solid rgba(148,163,184,0.35);
  font-weight: 700;
  letter-spacing: 0.04em;
}
.vraven-edge-inspector .status.ok { color: #32D74B; }
.vraven-edge-inspector .status.warn { color: #FF8A00; }
.vraven-edge-inspector .foot {
  margin-top: 10px;
  font-size: 11px;
  opacity: 0.75;
}
.vraven-edge-inspector .ladder {
  display: grid;
  gap: 3px;
  margin: 6px 0 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 11px;
}
.vraven-edge-inspector .ladder .ok { color: #32D74B; }
.vraven-edge-inspector .ladder .no { opacity: 0.55; }
.vraven-edge-inspector .layer-head {
  margin: 14px 0 0;
  padding: 6px 8px;
  border-radius: 5px;
  border: 1px solid rgba(148,163,184,0.35);
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  opacity: 0.8;
}
.vraven-edge-inspector .gaps { margin-top: 8px; }
.vraven-edge-inspector .gaps div { opacity: 0.75; margin: 2px 0; }
.vraven-edge-inspector .empty-mode {
  white-space: pre-wrap;
  opacity: 0.85;
  line-height: 1.45;
  padding-top: 8px;
}
.vraven-edge-inspector .edge-link {
  display: block;
  width: 100%;
  text-align: left;
  margin: 3px 0;
  padding: 4px 6px;
  border: 1px solid rgba(148,163,184,0.35);
  border-radius: 4px;
  background: transparent;
  color: inherit;
  cursor: pointer;
  font-size: 11px;
}
"""


def _replace_style(path: Path, style: str) -> bool:
    text = path.read_text(encoding="utf-8")
    updated, n = re.subn(
        r"<style>.*?</style>",
        f"<style>\n{style}\n</style>",
        text,
        count=1,
        flags=re.S,
    )
    if not n:
        return False
    path.write_text(updated, encoding="utf-8")
    return True


def patch_decision_proofs() -> int:
    count = 0
    for path in OUT.rglob("vraven_decision_proof.html"):
        if _replace_style(path, DECISION_STYLE):
            count += 1
            print(f"  patched decision proof · {path.relative_to(OUT)}")
    return count


def patch_proof_reports() -> int:
    count = 0
    for path in OUT.rglob("vraven_proof.html"):
        text = path.read_text(encoding="utf-8")
        if "Polished report chrome" in text:
            continue
        if "</style>" not in text:
            continue
        text = text.replace("</style>", PROOF_EXTRA_CSS + "\n</style>", 1)
        path.write_text(text, encoding="utf-8")
        count += 1
        print(f"  patched proof report · {path.relative_to(OUT)}")
    return count


def main() -> int:
    if not OUT.exists():
        print(f"Missing {OUT}")
        return 1
    print("Polishing Decision Proof HTML…")
    n_dec = patch_decision_proofs()
    print("Polishing Proof report HTML…")
    n_proof = patch_proof_reports()
    print("Rebuilding gallery pages…")
    from build_vraven_gallery import main as build_gallery

    build_gallery()
    print(f"Done. decision_proof={n_dec} proof={n_proof}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
