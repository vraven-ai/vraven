from __future__ import annotations

from .primitives import Canvas, alpha, regular_polygon
from .theme import Theme


def system_architecture(theme: Theme) -> Canvas:
    c = Canvas(1800, 1050, theme)

    # Input sources
    input_x, box_w = 55, 270
    c.box_label(input_x, 105, box_w, 102, ["Live PyTorch", "nn.Module"], accent=theme.support, sublabel="native deep mode")
    c.box_label(input_x, 235, box_w, 102, [".pt / .pth", "state dictionary"], accent=theme.stable, sublabel="model factory required")
    c.box_label(input_x, 365, box_w, 102, ["TorchScript", "module"], accent=theme.latent, sublabel="graph-aware mode")
    c.box_label(input_x, 520, box_w, 88, ["ONNX adapter"], accent=theme.neutral, sublabel="planned", dashed=True)
    c.box_label(input_x, 630, box_w, 88, ["TensorFlow adapter"], accent=theme.neutral, sublabel="planned", dashed=True)

    # Adapter gateway
    gate_x, gate_y, gate_w, gate_h = 395, 220, 285, 330
    c.rect(gate_x, gate_y, gate_w, gate_h, fill=theme.panel_alt, stroke=theme.border, radius=26)
    c.text("Model adapter", gate_x + gate_w / 2, gate_y + 50, size=25, weight=750, anchor="middle")
    c.pill(gate_x + 42, gate_y + 86, 200, 44, "device discovery", colour=theme.stable)
    c.pill(gate_x + 42, gate_y + 145, 200, 44, "nested input handling", colour=theme.support)
    c.pill(gate_x + 42, gate_y + 204, 200, 44, "output readout", colour=theme.latent)
    c.pill(gate_x + 42, gate_y + 263, 200, 44, "training-state restore", colour=theme.uncertainty)

    for y in (156, 286, 416, 564, 674):
        colour = theme.neutral if y > 500 else theme.text_secondary
        dash = "7 6" if y > 500 else None
        c.arrow(input_x + box_w + 6, y, gate_x - 10, gate_y + gate_h / 2, colour=colour, width=2.0, dash=dash, bend=(y - 385) * 0.11)

    # Instrumentation engine
    inst_x, inst_y, inst_w, inst_h = 755, 90, 300, 470
    c.rect(inst_x, inst_y, inst_w, inst_h, fill=theme.panel, stroke=theme.border, radius=26)
    c.text("Instrumentation", inst_x + inst_w/2, inst_y+48, size=25, weight=750, anchor="middle")
    inst_items = [
        ("module selection", theme.support),
        ("forward hooks", theme.support_strong),
        ("target score", theme.latent),
        ("optional gradients", theme.uncertainty),
        ("bounded interventions", theme.oppose),
    ]
    for i, (label, colour) in enumerate(inst_items):
        yy = inst_y + 82 + i*72
        c.circle(inst_x+42, yy+19, 12, fill=alpha(theme, colour, 0.18), stroke=colour, stroke_width=2)
        c.line(inst_x+42, yy+31, inst_x+42, yy+61 if i < len(inst_items)-1 else yy+31, stroke=theme.grid, width=2)
        c.text(label, inst_x+72, yy+25, size=19, weight=650)
        c.text(
            ["scoped and deterministic", "captures layer outputs", "class or scalar readout", "one autograd pass", "temporary ablation hooks"][i],
            inst_x+72, yy+50, size=13.5, colour=theme.text_secondary, weight=450,
        )
    c.arrow(gate_x+gate_w+10, gate_y+gate_h/2, inst_x-12, inst_y+inst_h/2, colour=theme.selected, width=3.0)

    # On-device reduction
    red_x, red_y, red_w, red_h = 1125, 70, 315, 510
    c.rect(red_x, red_y, red_w, red_h, fill=theme.panel_alt, stroke=theme.border, radius=26)
    c.text("On-device reduction", red_x+red_w/2, red_y+48, size=25, weight=750, anchor="middle")
    reduce_items = [
        ("activation statistics", "μ, σ², energy, sparsity", theme.support),
        ("signed interaction", "a ⊙ ∂sᵧ/∂a", theme.oppose),
        ("fixed-width profiles", "deterministic resampling", theme.latent),
        ("attention summaries", "weights kept distinct", theme.uncertainty),
        ("timing and shape", "bounded metadata", theme.stable),
    ]
    for i, (label, sub, colour) in enumerate(reduce_items):
        yy = red_y + 82 + i*78
        c.rect(red_x+28, yy, red_w-56, 58, fill=alpha(theme, colour, 0.08), stroke=alpha(theme, colour, 0.55), stroke_width=1.2, radius=12)
        c.rect(red_x+28, yy, 7, 58, fill=colour, stroke=colour, stroke_width=0, radius=5)
        c.text(label, red_x+51, yy+24, size=17.5, weight=650)
        c.text(sub, red_x+51, yy+46, size=13.2, colour=theme.text_secondary, weight=450)
    c.arrow(inst_x+inst_w+10, inst_y+inst_h/2, red_x-12, red_y+red_h/2, colour=theme.selected, width=3.0)

    # Unified trace core
    trace_cx, trace_cy = 1265, 750
    outer = regular_polygon(trace_cx, trace_cy, 150, 6)
    inner = regular_polygon(trace_cx, trace_cy, 118, 6)
    c.polygon(outer, fill=alpha(theme, theme.selected, 0.09), stroke=theme.selected, width=2.5)
    c.polygon(inner, fill=theme.panel, stroke=alpha(theme, theme.selected, 0.72), width=1.5)
    c.text("NeuralTrace", trace_cx, trace_cy-21, size=29, weight=800, anchor="middle")
    c.text("ordered layer records", trace_cx, trace_cy+14, size=16, colour=theme.text_secondary, anchor="middle")
    c.text("prediction • input • runtime", trace_cx, trace_cy+40, size=15, colour=theme.text_secondary, anchor="middle")
    c.arrow(red_x+red_w/2, red_y+red_h+12, trace_cx, trace_cy-165, colour=theme.selected, width=3.0)

    # Reasoning analyses around the trace
    analyses = [
        ("Pathways", 930, 665, theme.support),
        ("Fingerprints", 910, 825, theme.latent),
        ("Communities", 1510, 665, theme.stable),
        ("Uncertainty", 1515, 825, theme.uncertainty),
    ]
    for label, x, y, colour in analyses:
        c.box_label(x, y, 220, 76, label, accent=colour, fill=theme.panel, text_size=18)
        if x < trace_cx:
            c.arrow(trace_cx-122, trace_cy+(y-trace_cy)*0.28, x+220+8, y+38, colour=colour, width=2.1)
        else:
            c.arrow(trace_cx+122, trace_cy+(y-trace_cy)*0.28, x-8, y+38, colour=colour, width=2.1)

    # Outputs on right
    out_x, out_w = 1530, 220
    outputs = [
        ("Standalone figures", "SVG • PDF • PNG", theme.support),
        ("Interactive report", "local HTML", theme.selected),
        ("Trace persistence", "JSON schema", theme.latent),
    ]
    for i, (label, sub, colour) in enumerate(outputs):
        yy = 130+i*145
        c.box_label(out_x, yy, out_w, 104, label, accent=colour, sublabel=sub, text_size=18)
        c.arrow(red_x+red_w+12, red_y+110+i*120, out_x-12, yy+52, colour=colour, width=2.2)

    # Device lane
    lane_y = 960
    c.line(385, lane_y, 1450, lane_y, stroke=theme.grid, width=1.5)
    c.text("execution follows model device", 405, lane_y-25, size=15, colour=theme.text_secondary, weight=600)
    for i, (label, colour, dashed) in enumerate([
        ("CPU", theme.stable, False),
        ("CUDA", theme.support, False),
        ("MPS", theme.latent, False),
        ("ROCm", theme.neutral, True),
    ]):
        c.pill(705+i*155, lane_y-50, 120, 40, label, colour=colour, fill_strength=0.10)
        if dashed:
            # subtle planned / framework-path marker
            c.text("framework path", 765+i*155, lane_y+19, size=11.5, colour=theme.text_secondary, anchor="middle")
    c.text("large tensors remain transient; compact summaries cross to the host", 930, lane_y+48, size=14.5, colour=theme.text_secondary, anchor="middle")

    return c


def runtime_dataflow(theme: Theme) -> Canvas:
    c = Canvas(1800, 1000, theme)

    # Timeline spine
    x_positions = [110, 360, 650, 940, 1230, 1540]
    labels = ["Input", "Forward", "Target score", "Gradient", "Intervention", "Report"]
    colours = [theme.neutral, theme.support, theme.latent, theme.uncertainty, theme.oppose, theme.selected]
    for i, (x, label, colour) in enumerate(zip(x_positions, labels, colours, strict=False)):
        c.circle(x, 155, 26, fill=alpha(theme, colour, 0.16), stroke=colour, stroke_width=2.5, filter_id="soft_glow" if theme.name=="interactive_dark" and label in {"Forward","Report"} else None)
        c.text(label, x, 208, size=18, weight=700, anchor="middle")
        if i < len(x_positions)-1:
            c.arrow(x+34, 155, x_positions[i+1]-38, 155, colour=theme.text_secondary, width=2.3)

    c.pill(292, 84, 136, 38, "FAST", colour=theme.stable)
    c.line(110, 113, 650, 113, stroke=theme.stable, width=4.0, opacity=0.75)
    c.pill(785, 84, 168, 38, "GRADIENT", colour=theme.uncertainty)
    c.line(110, 100, 940, 100, stroke=theme.uncertainty, width=2.0, opacity=0.5, dash="7 6")
    c.pill(1138, 84, 146, 38, "CAUSAL", colour=theme.oppose)
    c.line(110, 87, 1230, 87, stroke=theme.oppose, width=2.0, opacity=0.45, dash="4 6")

    # Signal lanes
    lanes = [
        ("model tensors", 320, theme.support),
        ("captured signals", 465, theme.latent),
        ("persistent trace", 610, theme.stable),
        ("visual and report", 755, theme.selected),
    ]
    for label, y, colour in lanes:
        c.text(label, 78, y+4, size=17, colour=colour, weight=700)
        c.line(250, y, 1710, y, stroke=theme.grid, width=1.2)

    # Model tensor lane
    c.layer_stack(315, 278, 165, 70, 4, colour=theme.support, label="activations")
    c.arrow(500, 320, 640, 320, colour=theme.support, width=3.0)
    c.box_label(660, 278, 180, 82, "logits", accent=theme.latent, sublabel="primary output")
    c.arrow(860, 320, 925, 320, colour=theme.uncertainty, width=2.6)
    c.box_label(945, 278, 205, 82, ["selected score", "sᵧ"], accent=theme.uncertainty, sublabel="batch reduction")
    c.arrow(1170, 320, 1218, 320, colour=theme.oppose, width=2.6, dash="7 5")
    c.box_label(1240, 278, 190, 82, "ablation", accent=theme.oppose, sublabel="optional and bounded", dashed=True)
    c.arrow(1450, 320, 1580, 320, colour=theme.text_secondary, width=2.0)
    c.box_label(1600, 278, 120, 82, "output", accent=theme.selected, text_size=17)

    # Capture lane with branches
    capture_items = [
        (390, "μ, σ², E, S", theme.support),
        (710, "unit profile", theme.latent),
        (1010, "a ⊙ ∂sᵧ/∂a", theme.uncertainty),
        (1320, "Δsᵧ under ablation", theme.oppose),
    ]
    for x, label, colour in capture_items:
        c.box_label(x-100, 424, 200, 82, label, accent=colour, fill=alpha(theme, colour, 0.05), text_size=17)
    c.arrow(397, 360, 397, 414, colour=theme.support, width=2.0)
    c.arrow(755, 360, 755, 414, colour=theme.latent, width=2.0)
    c.arrow(1048, 360, 1048, 414, colour=theme.uncertainty, width=2.0)
    c.arrow(1335, 360, 1335, 414, colour=theme.oppose, width=2.0)

    # Persistent lane
    c.box_label(330, 565, 250, 90, ["LayerTrace", "records"], accent=theme.support, sublabel="ordered and compact")
    c.arrow(590, 610, 760, 610, colour=theme.selected, width=3.0)
    c.box_label(785, 548, 270, 124, ["NeuralTrace", "schema"], accent=theme.selected, sublabel="prediction • input • runtime", text_size=23)
    c.arrow(1068, 610, 1240, 610, colour=theme.selected, width=3.0)
    c.box_label(1260, 565, 250, 90, ["reasoning", "objects"], accent=theme.latent, sublabel="paths • DNA • communities")

    # Memory annotation
    c.rect(300, 690, 1215, 80, fill=theme.panel_alt, stroke=theme.border, radius=16)
    c.text("transient", 340, 722, size=15, colour=theme.support, weight=700)
    c.text("large activation tensors remain on the model device and are released after reduction", 435, 722, size=15, colour=theme.text_secondary)
    c.text("persistent", 340, 750, size=15, colour=theme.stable, weight=700)
    c.text("bounded layer summaries scale with captured layers and profile width, not raw activation volume", 435, 750, size=15, colour=theme.text_secondary)

    # Render lane
    render_items = [
        (360, "Decision flow", theme.support),
        (690, "Reasoning DNA", theme.latent),
        (1020, "Conflict field", theme.uncertainty),
        (1350, "Causal pathway", theme.oppose),
    ]
    for x, label, colour in render_items:
        c.box_label(x-110, 812, 220, 86, label, accent=colour, fill=theme.panel, text_size=17)
        c.arrow(920, 672, x, 802, colour=colour, width=1.8, opacity=0.8, bend=-20 if x<900 else 20)
    c.text("same measured trace", 890, 942, size=15, colour=theme.text_secondary, anchor="middle")
    c.pill(675, 916, 205, 42, "interactive_dark", colour=theme.selected)
    c.pill(900, 916, 220, 42, "publication_light", colour=theme.support)

    return c


def package_architecture(theme: Theme) -> Canvas:
    c = Canvas(1700, 1050, theme)

    # Public API top
    c.box_label(565, 65, 570, 90, ["Python public API", "explain • compare • monitor • render"], accent=theme.selected, sublabel="small stable surface", text_size=22)

    # Adapter row
    adapter_labels = [
        ("PyTorch", theme.support, False),
        ("TensorFlow", theme.neutral, True),
        ("ONNX", theme.neutral, True),
        ("Custom adapter", theme.latent, False),
    ]
    for i, (label, colour, dashed) in enumerate(adapter_labels):
        x = 120+i*380
        c.box_label(x, 220, 300, 90, label, accent=colour, dashed=dashed, sublabel="native" if label=="PyTorch" else ("plugin" if label=="Custom adapter" else "planned"), text_size=20)
        c.arrow(850, 165, x+150, 208, colour=colour, width=1.8, dash="7 6" if dashed else None, bend=-20 if i<2 else 20)

    # Layered core bands
    bands = [
        (365, 130, 1430, "capture", "hooks • selectors • tensor reduction", theme.support),
        (505, 220, 1250, "core", "configuration • trace schema • device • provenance", theme.selected),
        (645, 310, 1070, "metrics and reasoning", "influence • uncertainty • stability • pathways • fingerprints • communities", theme.latent),
        (785, 410, 870, "report and visualisation", "standalone figures • HTML • JSON • publication export", theme.stable),
    ]
    for y, x, w, label, sub, colour in bands:
        c.rect(x, y, w, 100, fill=alpha(theme, colour, 0.055), stroke=alpha(theme, colour, 0.7), stroke_width=2.0, radius=24)
        c.rect(x, y, 10, 100, fill=colour, stroke=colour, stroke_width=0, radius=7)
        c.text(label, x+35, y+39, size=22, weight=750, colour=colour)
        c.text(sub, x+35, y+71, size=15.5, colour=theme.text_secondary)

    # Optional acceleration sidecars
    c.box_label(75, 615, 250, 105, ["Rust core", "optional"], accent=theme.stable, sublabel="profiled CPU paths", dashed=True, text_size=20)
    c.box_label(1375, 615, 250, 105, ["CUDA / HIP", "optional kernels"], accent=theme.uncertainty, sublabel="fused reductions", dashed=True, text_size=20)
    c.arrow(335, 668, 530, 695, colour=theme.stable, width=2.0, dash="7 6")
    c.arrow(1365, 668, 1170, 695, colour=theme.uncertainty, width=2.0, dash="7 6")

    # Plugins and tests at base
    c.rect(130, 930, 1440, 70, fill=theme.panel_alt, stroke=theme.border, radius=18)
    base_items = [
        (250, "adapter registry", theme.support),
        (550, "metric plugins", theme.latent),
        (850, "visual plugins", theme.selected),
        (1150, "scientific tests", theme.stable),
        (1430, "release validation", theme.uncertainty),
    ]
    for x, label, colour in base_items:
        c.circle(x, 965, 8, fill=colour, stroke=colour, stroke_width=0)
        c.text(label, x+18, 966, size=15.5, colour=theme.text_secondary, baseline="middle")

    # Vertical flow
    c.arrow(850, 310, 850, 352, colour=theme.selected, width=3.0)
    c.arrow(850, 465, 850, 492, colour=theme.selected, width=3.0)
    c.arrow(850, 605, 850, 632, colour=theme.selected, width=3.0)
    c.arrow(850, 745, 850, 772, colour=theme.selected, width=3.0)
    c.arrow(850, 885, 850, 918, colour=theme.selected, width=3.0)

    return c


def device_execution(theme: Theme) -> Canvas:
    c = Canvas(1800, 1030, theme)

    # Left: invariant model state
    c.box_label(60, 395, 270, 180, ["User model", "and input"], accent=theme.selected, sublabel="device is discovered, not reassigned", text_size=24)

    lanes = [
        ("CPU", theme.stable, False, "vectorised tensor ops"),
        ("CUDA", theme.support, False, "GPU-side reduction"),
        ("Apple MPS", theme.latent, False, "Metal-backed tensors"),
        ("ROCm", theme.neutral, True, "framework path / validate"),
    ]
    for i, (label, colour, dashed, sub) in enumerate(lanes):
        y = 90+i*220
        # lane panel
        c.rect(420, y, 1310, 172, fill=theme.panel if i%2==0 else theme.panel_alt, stroke=theme.border, radius=20, dash="8 6" if dashed else None)
        c.pill(445, y+24, 170, 42, label, colour=colour)
        c.text(sub, 445, y+103, size=15.5, colour=theme.text_secondary)

        # model tensor reservoir
        c.layer_stack(670, y+50, 155, 58, 3, colour=colour, label="tensor")
        c.arrow(855, y+83, 990, y+83, colour=colour, width=3.0, dash="7 6" if dashed else None)
        c.box_label(1010, y+40, 250, 90, ["reduce in place", "or on device"], accent=colour, sublabel="statistics • profiles • score", text_size=18, dashed=dashed)
        c.arrow(1280, y+83, 1425, y+83, colour=theme.selected, width=2.8, dash="7 6" if dashed else None)
        c.box_label(1445, y+40, 245, 90, ["compact CPU", "summary"], accent=theme.selected, sublabel="bounded transfer", text_size=18)

        c.arrow(342, 485, 405, y+83, colour=colour, width=2.0, dash="7 6" if dashed else None, bend=(y+83-485)*0.10)

    # Crossed-out full tensor transfer warning
    c.rect(565, 925, 1040, 72, fill=alpha(theme, theme.oppose, 0.045), stroke=alpha(theme, theme.oppose, 0.55), radius=16)
    c.circle(610, 961, 18, fill="none", stroke=theme.oppose, stroke_width=2.5)
    c.line(598, 949, 622, 973, stroke=theme.oppose, width=2.5)
    c.text("no default full-activation copy to host memory", 650, 956, size=18, colour=theme.oppose, weight=700)
    c.text("full tensors are opt-in, scoped and explicitly released", 650, 980, size=14.5, colour=theme.text_secondary)

    return c
