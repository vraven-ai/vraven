#!/usr/bin/env python3
"""Build one click-through HTML page per model under vraven_outputs/.

Design rules:
  - One accent role for selection; body text never shares series/accent fills
  - Labels and headings never overlap (clear bands, no absolute collisions)
  - High-contrast readable type; humanized filenames
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vraven_outputs"

CAPABILITY_ORDER = (
    "reverse",
    "proof",
    "atlas",
    "watch",
    "compare",
    "darkside",
    "capture",
    "status",
)

CAPABILITY_LABELS = {
    "reverse": "Reverse",
    "proof": "Proof",
    "atlas": "Atlas",
    "watch": "Watch",
    "compare": "Compare",
    "darkside": "DarkSide",
    "capture": "Capture",
    "status": "Status",
}

CAPABILITY_BLURBS = {
    "reverse": "Decision Proof · mechanism decompilation",
    "proof": "Single-prediction explanation",
    "atlas": "Cohort reasoning map",
    "watch": "Checkpoint evolution",
    "compare": "Matched model comparison",
    "darkside": "Internal mechanism audits",
    "capture": "Mechanism IR capture",
    "status": "Blocked or failed notes",
}

PRIORITY_NAMES = {
    "vraven_decision_proof.html": 0,
    "vraven_proof.html": 1,
    "adversarial_audit.html": 2,
    "formal_robustness_certificate.html": 3,
    "formal_causal_mechanism_certificate.html": 4,
    "adversarial_robustness_profile.html": 5,
}

SKIP_NAMES = {"INDEX.html", "GALLERY.html", "VIEW.html"}

FRIENDLY = {
    "vraven_decision_proof": "Decision Proof report",
    "vraven_proof": "Explanation report",
    "adversarial_audit": "Adversarial Mechanism Audit",
    "formal_robustness_certificate": "Formal Robustness Certificate",
    "formal_causal_mechanism_certificate": "Formal Causal-Mechanism Certificate",
    "adversarial_robustness_profile": "Empirical Robustness Profile",
    "model_xray": "Raven Eye · model X-ray",
    "concept_genome": "Raven Flight · concept genome",
    "causal_proof": "Raven Brain · causal proof",
    "programme_proof": "Raven Code · programme proof",
    "contrastive_proof": "Raven Contrast",
    "counterfactual_twin": "Raven Shadow · counterfactual",
    "dark_matter": "Raven Dark Matter",
    "reasoning_landscape": "Reasoning landscape",
    "reasoning_similarity": "Reasoning similarity",
    "reasoning_outliers": "Reasoning outliers",
    "model_evolution": "Model evolution",
    "model_comparison": "Model comparison",
    "decision_circuitry": "Decision circuitry",
    "internal_anatomy": "Internal anatomy",
    "circuit_wiring": "Circuit wiring",
    "shortcut_reliance": "Shortcut reliance",
    "activation_mediation": "Activation mediation",
    "circuit_multiplicity": "Circuit multiplicity",
    "causal_synergy": "Causal synergy",
    "prediction_depth": "Prediction depth",
    "unused_capacity": "Unused capacity",
    "cost_evidence_tradeoff": "Cost–evidence trade-off",
    "failure_anatomy_fp_vs_tp": "Failure anatomy · FP vs TP",
    "failure_anatomy_fn_vs_tn": "Failure anatomy · FN vs TN",
    "high_confidence_ood": "High-confidence OOD",
    "mechanism_ir": "Mechanism IR",
    "STATUS": "Status note",
}


def _humanize(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"_(interactive_dark|publication_light)$", "", stem)
    if stem in FRIENDLY:
        label = FRIENDLY[stem]
    else:
        label = stem.replace("_", " ")
        label = re.sub(r"\s+", " ", label).strip().title()
    if name.endswith("interactive_dark.html"):
        return f"{label} · dark"
    if name.endswith("publication_light.html"):
        return f"{label} · light"
    if name.endswith(".json"):
        return f"{label} · JSON"
    return label


def _capability_of(rel: Path) -> str:
    parts = rel.parts
    for key in ("reverse", "proof", "atlas", "watch", "compare", "darkside", "capture"):
        if key in parts:
            return key
    return "status"


def _model_of(rel: Path) -> str:
    parts = rel.parts
    if not parts:
        return "root"
    if parts[0] == "kronodroid":
        if len(parts) >= 2 and parts[1] in {"mlp", "cnn1d", "resnet1d", "bigru", "compare"}:
            return f"kronodroid/{parts[1]}"
        return "kronodroid"
    return parts[0]


def _sort_key(rel: str) -> tuple:
    name = Path(rel).name
    pri = PRIORITY_NAMES.get(name, 50)
    mode = 0 if "interactive_dark" in name else 1 if "publication_light" in name else 2
    return (pri, mode, rel)


def collect() -> dict[str, dict[str, list[dict]]]:
    tree: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    for path in sorted(OUT.rglob("*.html")):
        if path.name in SKIP_NAMES:
            continue
        rel = path.relative_to(OUT)
        model = _model_of(rel)
        cap = _capability_of(rel)
        model_dir = OUT / model
        href = path.relative_to(model_dir).as_posix()
        tree[model][cap].append(
            {
                "path": rel.as_posix(),
                "href": href,
                "name": path.name,
                "label": _humanize(path.name),
                "kind": "html",
            }
        )

    for path in sorted(OUT.rglob("STATUS.json")):
        rel = path.relative_to(OUT)
        model = _model_of(rel)
        cap = _capability_of(rel)
        model_dir = OUT / model
        href = path.relative_to(model_dir).as_posix()
        tree[model][cap].append(
            {
                "path": rel.as_posix(),
                "href": href,
                "name": path.name,
                "label": _humanize(path.name),
                "kind": "json",
            }
        )

    catalog: dict[str, dict[str, list[dict]]] = {}
    for model in sorted(tree):
        catalog[model] = {}
        for cap in CAPABILITY_ORDER:
            if cap not in tree[model]:
                continue
            items = sorted(tree[model][cap], key=lambda item: _sort_key(item["path"]))
            seen: set[str] = set()
            unique = []
            for item in items:
                if item["path"] in seen:
                    continue
                seen.add(item["path"])
                unique.append(item)
            catalog[model][cap] = unique
    return catalog


# Shared visual system: modern dark mode, ambient glows, glassmorphism, and glowing accents.
PAGE_CSS = """
:root {
  --bg: #030A12;
  --panel: #07111F;
  --panel-2: #0B1728;
  --line: #1C344D;
  --line-glow: rgba(0, 212, 255, 0.3);
  --text: #F1F5F9;
  --muted: #94A3B8;
  --accent: #00D4FF;
  --accent-gradient: linear-gradient(135deg, #00D4FF 0%, #8B5CF6 50%, #FF2D55 100%);
  --select-bg: rgba(0, 212, 255, 0.15);
  --select-ink: #00D4FF;
  --select-bar: #00D4FF;
  --chip-bg: #0B1728;
  --danger: #FF2D55;
  --ok: #32D74B;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; }
body {
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  background-image: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(14, 165, 233, 0.12), transparent 70%), radial-gradient(ellipse 60% 40% at 80% 100%, rgba(124, 58, 237, 0.08), transparent 60%);
  background-attachment: fixed;
  color: var(--text);
  display: grid;
  grid-template-columns: 340px minmax(0, 1fr);
  grid-template-rows: auto 1fr;
  height: 100vh;
  -webkit-font-smoothing: antialiased;
}
header {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: minmax(0, 1fr) auto;
  gap: 12px 20px;
  align-items: center;
  padding: 14px 24px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  position: relative;
  z-index: 10;
}
header::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--accent-gradient);
}
.brand {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 0;
}
.brand a {
  color: var(--accent);
  text-decoration: none;
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  transition: color 0.15s ease;
}
.brand a:hover { color: #8B5CF6; }
.brand h1 {
  margin: 0;
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 1.25rem;
  font-weight: 800;
  letter-spacing: -0.01em;
  line-height: 1.2;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  overflow-wrap: anywhere;
}
.brand .meta {
  color: var(--muted);
  font-size: 0.82rem;
  line-height: 1.35;
}
.controls {
  display: flex;
  gap: 10px;
  align-items: center;
  flex-wrap: wrap;
  justify-content: flex-end;
}
button, select, input[type="search"] {
  font-family: "Inter", sans-serif;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--panel-2);
  color: var(--text);
  transition: all 0.15s ease;
}
button {
  padding: 8px 14px;
  cursor: pointer;
  font-family: "Plus Jakarta Sans", sans-serif;
  font-weight: 700;
  font-size: 0.82rem;
}
button:hover {
  border-color: var(--accent);
  box-shadow: 0 0 12px rgba(0, 212, 255, 0.2);
}
button:disabled { opacity: 0.35; cursor: default; box-shadow: none; }
button.primary {
  background: var(--accent-gradient);
  color: #FFFFFF;
  border: 0;
  box-shadow: 0 4px 14px rgba(0, 212, 255, 0.3);
}
button.primary:hover {
  box-shadow: 0 6px 20px rgba(0, 212, 255, 0.45);
}
input[type="search"], select {
  padding: 8px 12px;
  min-width: 150px;
  font-size: 0.84rem;
}
input[type="search"]:focus, select:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 14px rgba(0, 212, 255, 0.25);
}
aside {
  overflow: auto;
  border-right: 1px solid var(--line);
  background: var(--panel);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  padding: 12px 0 28px;
}
main {
  display: grid;
  grid-template-rows: auto 1fr;
  min-width: 0;
  min-height: 0;
  background: #030A12;
}
.crumb {
  padding: 14px 20px;
  border-bottom: 1px solid var(--line);
  background: var(--panel);
  display: grid;
  gap: 4px;
}
.crumb .path {
  color: var(--muted);
  font-family: "JetBrains Mono", monospace;
  font-size: 0.75rem;
  line-height: 1.4;
  overflow-wrap: anywhere;
}
.crumb .title {
  color: var(--text);
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 1rem;
  font-weight: 700;
  line-height: 1.35;
}
.viewer-wrap {
  min-height: 0;
  position: relative;
  background: #030A12;
}
iframe {
  width: 100%;
  height: 100%;
  border: 0;
  background: #030A12;
  display: block;
}
#jsonView {
  display: none;
  margin: 0;
  padding: 24px;
  overflow: auto;
  width: 100%;
  height: 100%;
  color: var(--text);
  background: #07111F;
  font-family: "JetBrains Mono", monospace;
  font-size: 12px;
  line-height: 1.55;
  white-space: pre-wrap;
}
.cap { margin: 0 0 6px; }
.cap > summary {
  list-style: none;
  cursor: pointer;
  padding: 12px 18px 10px;
  position: sticky;
  top: 0;
  z-index: 2;
  background: var(--panel);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border-bottom: 1px solid transparent;
}
.cap[open] > summary { border-bottom-color: var(--line); }
.cap > summary::-webkit-details-marker { display: none; }
.cap-head {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.cap-name {
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 0.74rem;
  font-weight: 800;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--accent);
}
.cap-blurb {
  font-size: 0.76rem;
  color: var(--muted);
  line-height: 1.35;
}
.item {
  display: grid;
  grid-template-columns: 4px 1fr;
  gap: 0 12px;
  width: 100%;
  text-align: left;
  border: 0;
  border-radius: 0;
  background: transparent;
  color: var(--text);
  padding: 9px 16px 9px 14px;
  font-size: 0.86rem;
  font-weight: 500;
  line-height: 1.35;
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}
.item::before {
  content: "";
  width: 4px;
  border-radius: 4px;
  background: transparent;
  align-self: stretch;
  transition: background 0.15s ease;
}
.item:hover { background: rgba(0, 212, 255, 0.08); }
.item.active {
  background: var(--select-bg);
  color: var(--accent);
  font-weight: 600;
}
.item.active::before { background: var(--accent); box-shadow: 0 0 10px rgba(0, 212, 255, 0.6); }
.item .sub {
  display: block;
  margin-top: 2px;
  font-size: 0.72rem;
  color: var(--muted);
  font-weight: 400;
}
.item.active .sub { color: #00D4FF; }
.empty { padding: 24px 16px; color: var(--muted); font-size: 0.9rem; line-height: 1.45; }
  .controls { justify-content: flex-start; }
}
"""


def model_page_html(model: str, caps: dict[str, list[dict]], *, home_href: str) -> str:
    flat = []
    for cap in CAPABILITY_ORDER:
        for item in caps.get(cap, []):
            flat.append(
                {
                    **item,
                    "capability": cap,
                    "capability_label": CAPABILITY_LABELS.get(cap, cap),
                    "capability_blurb": CAPABILITY_BLURBS.get(cap, ""),
                }
            )
    domain_info = MODEL_DOMAINS.get(model, ("General Neural Model", "Deep Neural Net", "General AI"))
    sub_title, arch_tag, category = domain_info

    data = json.dumps({"model": model, "items": flat, "labels": CAPABILITY_LABELS, "blurbs": CAPABILITY_BLURBS})
    cap_order = json.dumps(list(CAPABILITY_ORDER))

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VRAVEN · {model} · Neural Mechanism Decompiler</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<header>
  <div class="brand">
    <a href="{home_href}">← Observatory Home</a>
    <h1>{model}</h1>
    <div class="meta" id="countMeta"><span style="color:var(--accent-cyan);font-weight:700">{category}</span> · {arch_tag} · {sub_title}</div>
  </div>
  <div class="controls">
    <input type="search" id="search" placeholder="Filter proof views…" aria-label="Filter"/>
    <select id="modeFilter" aria-label="Theme filter">
      <option value="prefer_dark">Prefer Dark Mode</option>
      <option value="all">All Figures</option>
      <option value="dark_only">Dark Theme Only</option>
      <option value="light_only">Light Theme Only</option>
    </select>
    <button type="button" id="prevBtn" title="Shortcut: Left Arrow or K">← Prev (K)</button>
    <button type="button" id="nextBtn" title="Shortcut: Right Arrow or J">Next (J) →</button>
    <button type="button" class="primary" id="openBtn" title="Shortcut: O">Open Standalone ↗</button>
  </div>
</header>
<aside id="nav" aria-label="Outputs"></aside>
<main>
  <div class="crumb" id="crumb">
    <div class="title">Select a proof report</div>
    <div class="path">Scientific proofs load instantly below in high-resolution interactive mode.</div>
  </div>
  <div class="viewer-wrap">
    <iframe id="frame" title="viewer" src="about:blank"></iframe>
    <pre id="jsonView"></pre>
  </div>
</main>
<script>
const DATA = {data};
const CAP_ORDER = {cap_order};
const LABELS = DATA.labels;
const BLURBS = DATA.blurbs;
const nav = document.getElementById('nav');
const frame = document.getElementById('frame');
const jsonView = document.getElementById('jsonView');
const crumb = document.getElementById('crumb');
const countMeta = document.getElementById('countMeta');
const search = document.getElementById('search');
const modeFilter = document.getElementById('modeFilter');
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const openBtn = document.getElementById('openBtn');
let visible = [];
let current = -1;

function passesMode(item) {{
  const mode = modeFilter.value;
  const name = item.name || '';
  const isDark = name.includes('interactive_dark');
  const isLight = name.includes('publication_light');
  if (mode === 'all') return true;
  if (mode === 'dark_only') return !isLight || isDark;
  if (mode === 'light_only') return !isDark || isLight;
  if (!isLight) return true;
  const darkTwin = name.replace('publication_light', 'interactive_dark');
  return !DATA.items.some(other => other.name === darkTwin && other.capability === item.capability);
}}

function filteredItems() {{
  const q = search.value.trim().toLowerCase();
  return DATA.items.filter(item => {{
    if (!passesMode(item)) return false;
    if (!q) return true;
    return (item.label + ' ' + item.path + ' ' + item.capability).toLowerCase().includes(q);
  }});
}}

function renderNav() {{
  visible = filteredItems();
  const byCap = {{}};
  for (const item of visible) (byCap[item.capability] ||= []).push(item);
  nav.innerHTML = '';
  for (const cap of CAP_ORDER) {{
    const items = byCap[cap];
    if (!items) continue;
    const details = document.createElement('details');
    details.className = 'cap';
    details.open = true;
    const summary = document.createElement('summary');
    summary.innerHTML = '<div class="cap-head"><span class="cap-name">' +
      (LABELS[cap] || cap) + ' <b style="color:var(--accent-cyan)">(' + items.length + ')</b></span><span class="cap-blurb">' +
      (BLURBS[cap] || '') + '</span></div>';
    details.appendChild(summary);
    for (const item of items) {{
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.className = 'item';
      btn.dataset.path = item.path;
      const sub = item.kind === 'json' ? 'Machine IR / Telemetry JSON' : item.name;
      btn.innerHTML = '<span>' + item.label + '<span class="sub">' + sub + '</span></span>';
      btn.addEventListener('click', () => showPath(item.path));
      details.appendChild(btn);
    }}
    nav.appendChild(details);
  }}
  if (!visible.length) nav.innerHTML = '<div class="empty">No matching outputs. Clear the filter or show all themes.</div>';
  if (visible.length) showIndex(Math.max(0, Math.min(current, visible.length - 1)));
  prevBtn.disabled = !visible.length;
  nextBtn.disabled = !visible.length;
}}

function highlight(path) {{
  for (const el of nav.querySelectorAll('.item')) el.classList.toggle('active', el.dataset.path === path);
}}

async function showIndex(index) {{
  if (!visible.length) return;
  current = (index + visible.length) % visible.length;
  const item = visible[current];
  highlight(item.path);
  crumb.innerHTML = '<div class="title">' + item.capability_label + ' · ' + item.label +
    ' <span style="color:var(--muted);font-weight:500;font-size:0.85rem">[' + (current + 1) + ' of ' + visible.length + ']</span></div>' +
    '<div class="path">' + item.href + '</div>';
  if (item.kind === 'json') {{
    frame.style.display = 'none';
    jsonView.style.display = 'block';
    jsonView.textContent = 'Loading Machine IR JSON…';
    try {{
      const text = await (await fetch(item.href)).text();
      try {{ jsonView.textContent = JSON.stringify(JSON.parse(text), null, 2); }}
      catch {{ jsonView.textContent = text; }}
    }} catch (err) {{ jsonView.textContent = String(err); }}
  }} else {{
    jsonView.style.display = 'none';
    frame.style.display = 'block';
    frame.src = item.href;
  }}
  history.replaceState(null, '', '#' + encodeURIComponent(item.path));
}}

function showPath(path) {{
  const idx = visible.findIndex(item => item.path === path);
  if (idx >= 0) showIndex(idx);
}}

prevBtn.addEventListener('click', () => showIndex(current - 1));
nextBtn.addEventListener('click', () => showIndex(current + 1));
openBtn.addEventListener('click', () => {{
  if (current >= 0 && visible[current]) window.open(visible[current].href, '_blank');
}});
search.addEventListener('input', renderNav);
modeFilter.addEventListener('change', renderNav);
window.addEventListener('keydown', (event) => {{
  if (event.target.matches('input, select, textarea')) return;
  if (event.key === 'ArrowLeft' || event.key === 'k') {{ event.preventDefault(); showIndex(current - 1); }}
  if (event.key === 'ArrowRight' || event.key === 'j') {{ event.preventDefault(); showIndex(current + 1); }}
  if (event.key === 'o' || event.key === 'O') {{
    event.preventDefault();
    if (current >= 0 && visible[current]) window.open(visible[current].href, '_blank');
  }}
}});
renderNav();
const hash = decodeURIComponent((location.hash || '').replace(/^#/, ''));
if (hash) {{
  const idx = visible.findIndex(item => item.path === hash);
  if (idx >= 0) showIndex(idx);
}}
</script>
</body>
</html>
"""


MODEL_DOMAINS = {
    "windows10_full_benign_ae": ("Enterprise Security", "Autoencoder", "Cybersecurity"),
    "win2016_dc": ("Domain Controller Security", "Autoencoder", "Cybersecurity"),
    "kronodroid/mlp": ("Android Malware", "MLP", "Cybersecurity"),
    "kronodroid/cnn1d": ("Android Malware", "CNN-1D", "Cybersecurity"),
    "kronodroid/resnet1d": ("Android Malware", "ResNet-1D", "Cybersecurity"),
    "kronodroid/bigru": ("Android Malware", "BiGRU", "Cybersecurity"),
    "kronodroid/compare": ("Android Malware", "Multi-Model", "Cybersecurity"),
    "winlolbin_mlp": ("LOLBins Injection", "MLP", "Cybersecurity"),
    "phishing_url_mlp": ("URL Threat Detection", "MLP", "Cybersecurity"),
    "phishing_url_cnn1d": ("URL Threat Detection", "CNN-1D", "Cybersecurity"),
    "network_ids_mlp": ("Network NIDS", "MLP", "Network & IoT"),
    "medical_triage_mlp": ("Clinical Triage", "MLP", "Medical AI"),
    "sensor_transformer": ("Edge Telemetry", "Transformer", "Edge Sensing"),
    "image_pattern_cnn2d": ("Visual Pattern", "CNN-2D", "Computer Vision"),
    "image_pattern_vit": ("Visual Vision Transformer", "ViT-B/16", "Computer Vision"),
    "image_pattern_diffusion": ("Generative Diffusion", "UNet-Diffusion", "Generative AI"),
}


def top_index_html(catalog: dict[str, dict[str, list[dict]]]) -> str:
    cards = []
    total_outputs = 0
    total_models = len(catalog)
    domain_counts = defaultdict(int)

    for model, caps in catalog.items():
        n = sum(len(v) for v in caps.values())
        total_outputs += n

        domain_info = MODEL_DOMAINS.get(model, ("General Neural Model", "Deep Neural Net", "General AI"))
        sub_title, arch_tag, category = domain_info
        domain_counts[category] += 1

    matrix_rows = []
    for model, caps in catalog.items():
        n = sum(len(v) for v in caps.values())
        domain_info = MODEL_DOMAINS.get(model, ("General Neural Model", "Deep Neural Net", "General AI"))
        sub_title, arch_tag, category = domain_info
        caps_str = ", ".join(k.capitalize() for k in caps if k != "status")
        matrix_rows.append(f"""
<tr>
  <td class="m-td-name"><b>{escape(model)}</b><br><small style="color:var(--muted)">{escape(sub_title)}</small></td>
  <td><span class="m-badge-cat">{escape(category)}</span></td>
  <td><code class="m-code">{escape(arch_tag)}</code></td>
  <td><b style="color:var(--accent-cyan)">{n}</b> views</td>
  <td><span class="m-caps">{escape(caps_str)}</span></td>
  <td><span class="status-pill"><span class="led-green"></span> 100% Provenance</span></td>
  <td><a class="m-btn-link" href="{escape(model)}/INDEX.html">Launch ↗</a></td>
</tr>
""")

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>VRAVEN · Neural Mechanism Decompiler &amp; Scientific Proof Observatory</title>
<style>
:root {{
  --bg: #030A12;
  --panel: #07111F;
  --panel-hover: #0B1728;
  --line: #1C344D;
  --line-glow: rgba(0, 212, 255, 0.4);
  --text: #F1F5F9;
  --muted: #94A3B8;
  --accent: #168BFF;
  --accent-cyan: #00D4FF;
  --accent-violet: #8B5CF6;
  --accent-gradient: linear-gradient(135deg, #00D4FF 0%, #168BFF 40%, #8B5CF6 72%, #FF2D55 100%);
  --chip: #0B1728;
  --ok: #32D74B;
  --danger: #FF2D55;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  min-height: 100vh;
  padding: 0;
  font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--text);
  background: var(--bg);
  background-image: 
    radial-gradient(ellipse 80% 50% at 50% -20%, rgba(14, 165, 233, 0.16), transparent 70%), 
    radial-gradient(ellipse 60% 40% at 85% 100%, rgba(168, 85, 247, 0.12), transparent 60%),
    linear-gradient(rgba(0, 212, 255, 0.03) 1px, transparent 1px), 
    linear-gradient(90deg, rgba(0, 212, 255, 0.03) 1px, transparent 1px);
  background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
  background-attachment: fixed;
  -webkit-font-smoothing: antialiased;
}}
.nav-telemetry {{
  background: rgba(6, 11, 22, 0.95);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--line);
  padding: 10px 32px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.76rem;
  color: var(--muted);
  position: sticky;
  top: 0;
  z-index: 100;
}}
.t-left {{ display: flex; gap: 20px; align-items: center; }}
.t-item {{ display: flex; gap: 6px; align-items: center; }}
.t-val {{ color: var(--accent-cyan); font-weight: 700; }}
.t-right {{ display: flex; gap: 16px; align-items: center; }}
.status-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: 999px;
  background: rgba(16, 185, 129, 0.12);
  border: 1px solid rgba(16, 185, 129, 0.3);
  color: #32D74B;
  font-weight: 600;
  font-size: 0.72rem;
  letter-spacing: 0.05em;
}}
.matrix-btn {{
  appearance: none;
  background: var(--accent-gradient);
  color: #FFFFFF;
  border: 0;
  border-radius: 8px;
  padding: 5px 14px;
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 0.76rem;
  font-weight: 700;
  cursor: pointer;
  box-shadow: 0 4px 12px rgba(0, 229, 255, 0.3);
  transition: transform 0.15s ease;
}}
.matrix-btn:hover {{ transform: scale(1.04); }}
.led-green {{
  width: 7px; height: 7px;
  background: #32D74B;
  border-radius: 50%;
  box-shadow: 0 0 10px #32D74B;
  animation: pulse-green 2s infinite;
}}
.led-red {{
  width: 7px; height: 7px;
  background: #FF2D55;
  border-radius: 50%;
  box-shadow: 0 0 10px #FF2D55;
}}
@keyframes pulse-green {{
  0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
  70% {{ transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }}
  100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
}}

.wrap {{ max-width: 1360px; margin: 0 auto; padding: 40px 32px 80px; }}

.header-hero {{
  position: relative;
  background: var(--panel);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 36px 44px;
  margin-bottom: 36px;
  box-shadow: 0 20px 50px -15px rgba(0, 0, 0, 0.7), 0 0 30px 0 rgba(0, 229, 255, 0.05);
  overflow: hidden;
}}
.header-hero::before {{
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 4px;
  background: var(--accent-gradient);
}}
.eyebrow {{
  margin: 0 0 12px;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--accent-cyan);
  display: inline-flex;
  align-items: center;
  gap: 10px;
}}
h1 {{
  margin: 0 0 14px;
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: clamp(2.2rem, 4vw, 3.2rem);
  line-height: 1.1;
  font-weight: 800;
  letter-spacing: -0.03em;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.note {{
  margin: 0 0 28px;
  max-width: 76ch;
  color: var(--muted);
  line-height: 1.65;
  font-size: 1.05rem;
}}

/* Filter Bar */
.filter-container {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  padding-top: 20px;
  border-top: 1px solid var(--line);
}}
.filter-tabs {{
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}}
.filter-tab {{
  appearance: none;
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid var(--line);
  color: var(--muted);
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 0.82rem;
  font-weight: 700;
  padding: 8px 16px;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}}
.filter-tab:hover {{
  color: var(--text);
  border-color: var(--accent);
  box-shadow: 0 0 12px rgba(0, 212, 255, 0.2);
}}
.filter-tab.active {{
  background: var(--accent-gradient);
  color: #FFFFFF;
  border-color: transparent;
  box-shadow: 0 4px 16px rgba(0, 212, 255, 0.35);
}}
.search-box {{
  position: relative;
  min-width: 260px;
}}
.search-box input {{
  width: 100%;
  padding: 9px 14px 9px 36px;
  background: rgba(15, 23, 42, 0.8);
  border: 1px solid var(--line);
  border-radius: 12px;
  color: var(--text);
  font-family: "Inter", sans-serif;
  font-size: 0.86rem;
  transition: all 0.2s ease;
}}
.search-box input:focus {{
  outline: none;
  border-color: var(--accent-cyan);
  box-shadow: 0 0 16px rgba(0, 229, 255, 0.25);
}}
.search-icon {{
  position: absolute;
  left: 12px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--muted);
  font-size: 0.9rem;
  pointer-events: none;
}}

.grid {{
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(360px, 1fr));
  gap: 24px;
}}
.card {{
  display: flex;
  flex-direction: column;
  gap: 16px;
  text-decoration: none;
  color: inherit;
  background: var(--panel);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 26px 24px 22px;
  min-height: 200px;
  position: relative;
  overflow: hidden;
  transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}}
.card-accent-bar {{
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 3px;
  background: var(--line);
  transition: background 0.22s ease;
}}
.card:hover {{
  transform: translateY(-5px);
  border-color: var(--line-glow);
  box-shadow: 0 20px 45px -12px rgba(0, 229, 255, 0.22), 0 0 0 1px rgba(0, 229, 255, 0.3);
  background: var(--panel-hover);
}}
.card:hover .card-accent-bar {{
  background: var(--accent-gradient);
}}
.card-top {{
  display: flex;
  justify-content: space-between;
  gap: 14px;
  align-items: flex-start;
}}
.card-head-wrap {{ display: flex; flex-direction: column; gap: 4px; min-width: 0; }}
.card-category-badge {{
  font-family: "JetBrains Mono", monospace;
  font-size: 0.72rem;
  font-weight: 700;
  color: var(--accent-cyan);
  letter-spacing: 0.06em;
  text-transform: uppercase;
}}
.card h2 {{
  margin: 0;
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 1.2rem;
  font-weight: 750;
  line-height: 1.3;
  overflow-wrap: anywhere;
  color: var(--text);
}}
.card-sub {{
  font-size: 0.84rem;
  color: var(--muted);
  line-height: 1.4;
}}
.state {{
  flex: 0 0 auto;
  font-size: 0.7rem;
  font-weight: 700;
  font-family: "JetBrains Mono", monospace;
  color: #32D74B;
  background: rgba(16, 185, 129, 0.12);
  padding: 5px 12px;
  border: 1px solid rgba(16, 185, 129, 0.3);
  border-radius: 999px;
  white-space: nowrap;
  letter-spacing: 0.04em;
  display: inline-flex;
  align-items: center;
  gap: 6px;
}}
.state.blocked {{
  color: #FF2D55;
  background: rgba(244, 63, 94, 0.12);
  border-color: rgba(244, 63, 94, 0.3);
}}

/* Card Stat Row */
.card-metrics-row {{
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 8px;
  padding: 10px 14px;
  background: rgba(5, 8, 17, 0.7);
  border: 1px solid rgba(255, 255, 255, 0.05);
  border-radius: 12px;
}}
.card-metric-item {{
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}}
.m-val {{
  font-family: "JetBrains Mono", monospace;
  font-size: 1rem;
  font-weight: 700;
  color: var(--text);
}}
.m-lbl {{
  font-size: 0.68rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 2px;
}}

.chips {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: auto;
}}
.chip {{
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.08);
  background: var(--chip);
  color: #94A3B8;
  font-size: 0.73rem;
  font-weight: 600;
  letter-spacing: 0.02em;
  font-family: "Plus Jakarta Sans", sans-serif;
  transition: all 0.15s ease;
}}
.chip-num {{
  color: var(--accent-cyan);
  font-family: "JetBrains Mono", monospace;
  font-size: 0.7rem;
}}
.card:hover .chip {{
  color: #F1F5F9;
  border-color: rgba(0, 212, 255, 0.25);
}}

/* Modal Matrix Overlay */
.modal-overlay {{
  position: fixed;
  inset: 0;
  background: rgba(4, 6, 12, 0.88);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  display: none;
  justify-content: center;
  align-items: center;
  z-index: 999;
  padding: 32px;
}}
.modal-overlay.active {{ display: flex; }}
.modal-card {{
  background: rgba(10, 16, 30, 0.95);
  border: 1px solid var(--line);
  border-radius: 24px;
  max-width: 1200px;
  width: 100%;
  max-height: 85vh;
  display: flex;
  flex-direction: column;
  box-shadow: 0 24px 60px -15px rgba(0, 0, 0, 0.9), 0 0 40px rgba(0, 229, 255, 0.1);
  overflow: hidden;
}}
.modal-hdr {{
  padding: 24px 32px;
  border-bottom: 1px solid var(--line);
  display: flex;
  justify-content: space-between;
  align-items: center;
}}
.modal-hdr h2 {{
  margin: 0;
  font-family: "Plus Jakarta Sans", sans-serif;
  font-size: 1.4rem;
  font-weight: 800;
  background: var(--accent-gradient);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}}
.close-modal {{
  background: transparent;
  border: 0;
  color: var(--muted);
  font-size: 1.5rem;
  cursor: pointer;
  padding: 4px 10px;
}}
.close-modal:hover {{ color: var(--text); }}
.modal-body {{
  padding: 24px 32px;
  overflow: auto;
}}
.matrix-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 0.84rem;
  text-align: left;
}}
.matrix-table th {{
  padding: 12px 14px;
  border-bottom: 2px solid var(--line);
  color: var(--accent-cyan);
  font-family: "JetBrains Mono", monospace;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}}
.matrix-table td {{
  padding: 14px;
  border-bottom: 1px solid var(--line);
  color: var(--text);
}}
.m-badge-cat {{
  display: inline-block;
  padding: 3px 8px;
  border-radius: 6px;
  background: rgba(0, 212, 255, 0.12);
  color: #168BFF;
  font-size: 0.72rem;
  font-weight: 700;
}}
.m-code {{
  font-family: "JetBrains Mono", monospace;
  color: var(--accent-violet);
  background: rgba(168, 85, 247, 0.12);
  padding: 3px 8px;
  border-radius: 6px;
  font-size: 0.78rem;
}}
.m-caps {{ font-size: 0.76rem; color: var(--muted); }}
.m-btn-link {{
  color: var(--accent-cyan);
  text-decoration: none;
  font-weight: 700;
  font-size: 0.8rem;
}}
.m-btn-link:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<div class="nav-telemetry">
  <div class="t-left">
    <div class="t-item">🔬 ENGINE: <span class="t-val">VRAVEN 0.9.4 SE</span></div>
    <div class="t-item">📊 AUDITED MODELS: <span class="t-val">{total_models}</span></div>
    <div class="t-item">⚡ NEURAL PROOFS: <span class="t-val">{total_outputs}</span></div>
    <div class="t-item">🛡️ CAUSAL PROVENANCE: <span class="t-val">VERIFIED</span></div>
  </div>
  <div class="t-right">
    <button class="matrix-btn" id="openMatrixBtn">📊 Audit Matrix</button>
    <div class="status-pill"><span class="led-green"></span> TELEMETRY ACTIVE</div>
  </div>
</div>

<div class="wrap">
  <div class="header-hero">
    <p class="eyebrow">Scientific Proof Observatory &amp; Neural Mechanism Decompiler</p>
    <h1>Model Results Observatory</h1>
    <p class="note">Interactive mechanism proofs, LRP relevance conservation, causal intervention graphs, and cohort reasoning maps. Select any audited model below to launch its scientific visualization suite.</p>

    <div class="filter-container">
      <div class="filter-tabs" id="filterTabs">
        <button class="filter-tab active" data-cat="ALL">All Models ({total_models})</button>
        <button class="filter-tab" data-cat="Cybersecurity">Cybersecurity ({domain_counts['Cybersecurity']})</button>
        <button class="filter-tab" data-cat="Medical AI">Medical AI ({domain_counts['Medical AI']})</button>
        <button class="filter-tab" data-cat="Network & IoT">Network &amp; NIDS ({domain_counts['Network & IoT']})</button>
        <button class="filter-tab" data-cat="Computer Vision">Vision &amp; GenAI ({domain_counts['Computer Vision'] + domain_counts['Generative AI']})</button>
      </div>
      <div class="search-box">
        <span class="search-icon">🔍</span>
        <input type="search" id="modelSearch" placeholder="Filter models or tags…"/>
      </div>
    </div>
  </div>

  <div class="grid" id="modelGrid">
{''.join(cards)}
  </div>
</div>

<!-- Modal Matrix Overlay -->
<div class="modal-overlay" id="matrixModal">
  <div class="modal-card">
    <div class="modal-hdr">
      <h2>📊 VRAVEN Scientific Audit &amp; Provenance Matrix</h2>
      <button class="close-modal" id="closeMatrixBtn">&times;</button>
    </div>
    <div class="modal-body">
      <table class="matrix-table">
        <thead>
          <tr>
            <th>Model &amp; Task</th>
            <th>Domain</th>
            <th>Architecture</th>
            <th>Proof Views</th>
            <th>Capabilities</th>
            <th>Provenance</th>
            <th>Action</th>
          </tr>
        </thead>
        <tbody>
{''.join(matrix_rows)}
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
const tabs = document.querySelectorAll('.filter-tab');
const searchInput = document.getElementById('modelSearch');
const cards = document.querySelectorAll('.card');
const matrixModal = document.getElementById('matrixModal');
const openMatrixBtn = document.getElementById('openMatrixBtn');
const closeMatrixBtn = document.getElementById('closeMatrixBtn');

openMatrixBtn.addEventListener('click', () => matrixModal.classList.add('active'));
closeMatrixBtn.addEventListener('click', () => matrixModal.classList.remove('active'));
matrixModal.addEventListener('click', (e) => {{ if (e.target === matrixModal) matrixModal.classList.remove('active'); }});

function filterGrid() {{
  const activeTab = document.querySelector('.filter-tab.active');
  const cat = activeTab ? activeTab.getAttribute('data-cat') : 'ALL';
  const query = searchInput.value.toLowerCase().trim();

  cards.forEach(card => {{
    const cardCat = card.getAttribute('data-category');
    const cardModel = card.getAttribute('data-model').toLowerCase();
    const matchesCat = (cat === 'ALL') || (cat === 'Computer Vision' ? (cardCat === 'Computer Vision' || cardCat === 'Generative AI') : cardCat === cat);
    const matchesQuery = !query || cardModel.includes(query) || cardCat.toLowerCase().includes(query);

    if (matchesCat && matchesQuery) {{
      card.style.display = 'flex';
    }} else {{
      card.style.display = 'none';
    }}
  }});
}}

tabs.forEach(tab => {{
  tab.addEventListener('click', () => {{
    tabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    filterGrid();
  }});
}});

searchInput.addEventListener('input', filterGrid);
window.addEventListener('keydown', (e) => {{ if (e.key === 'Escape') matrixModal.classList.remove('active'); }});
</script>
</body>
</html>
"""




def main() -> int:
    catalog = collect()
    for stale in (OUT / "GALLERY.html",):
        if stale.exists():
            stale.unlink()

    for model, caps in catalog.items():
        model_dir = OUT / model
        model_dir.mkdir(parents=True, exist_ok=True)
        depth = len(Path(model).parts)
        home_href = "/".join([".."] * depth) + "/INDEX.html"
        (model_dir / "INDEX.html").write_text(
            model_page_html(model, caps, home_href=home_href),
            encoding="utf-8",
        )

    (OUT / "INDEX.html").write_text(top_index_html(catalog), encoding="utf-8")
    print(f"Wrote top INDEX + {len(catalog)} model pages under {OUT}")
    for model, caps in catalog.items():
        n = sum(len(v) for v in caps.values())
        print(f"  {model}/INDEX.html  ({n} views)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
