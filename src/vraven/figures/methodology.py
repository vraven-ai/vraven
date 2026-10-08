from __future__ import annotations

import math

from .primitives import Canvas, alpha, mix
from .theme import Theme


def explanation_modes(theme: Theme) -> Canvas:
    c = Canvas(1800, 1040, theme)

    cols = [
        (90, "FAST", theme.stable, "one forward pass"),
        (650, "GRADIENT", theme.uncertainty, "one forward + one backward"),
        (1210, "CAUSAL", theme.oppose, "gradient trace + bounded interventions"),
    ]
    for x, label, colour, sub in cols:
        c.rect(x, 70, 500, 880, fill=theme.panel if x != 650 else theme.panel_alt, stroke=theme.border, radius=24)
        c.pill(x+35, 100, 160 if label!="GRADIENT" else 190, 46, label, colour=colour)
        c.text(sub, x+35, 180, size=16, colour=theme.text_secondary, weight=600)

    # Fast
    fx = 90
    fast_steps = [
        ("model(x)", "inference_mode", theme.support),
        ("capture outputs", "selected modules", theme.support_strong),
        ("reduce tensors", "μ • σ² • energy • sparsity", theme.latent),
        ("build trace", "forward observational signals", theme.stable),
        ("render", "flow • DNA • spectral trace", theme.selected),
    ]
    for i, (lab, sub, colour) in enumerate(fast_steps):
        yy = 240+i*132
        c.box_label(fx+55, yy, 390, 88, lab, accent=colour, sublabel=sub, text_size=19)
        if i < len(fast_steps)-1:
            c.arrow(fx+250, yy+94, fx+250, yy+124, colour=colour, width=2.5)

    # Gradient
    gx = 650
    grad_steps = [
        ("model(x)", "gradients enabled", theme.support),
        ("select target score sᵧ", "class, scalar or output adapter", theme.latent),
        ("autograd.grad", "selected activations and inputs", theme.uncertainty),
        ("interaction", "a ⊙ ∂sᵧ/∂a", theme.oppose),
        ("target-specific trace", "signed observational evidence", theme.selected),
    ]
    for i, (lab, sub, colour) in enumerate(grad_steps):
        yy = 240+i*132
        c.box_label(gx+55, yy, 390, 88, lab, accent=colour, sublabel=sub, text_size=19)
        if i < len(grad_steps)-1:
            c.arrow(gx+250, yy+94, gx+250, yy+124, colour=colour, width=2.5)

    # Causal
    cx = 1210
    causal_steps = [
        ("rank candidates", "from gradient trace", theme.uncertainty),
        ("define intervention", "zero • scale • replace", theme.oppose),
        ("re-run inference", "hook scoped and reversible", theme.support),
        ("measure Δsᵧ", "with matched control", theme.oppose_strong),
        ("record causal support", "bounded, not mechanism recovery", theme.selected),
    ]
    for i, (lab, sub, colour) in enumerate(causal_steps):
        yy = 240+i*132
        c.box_label(cx+55, yy, 390, 88, lab, accent=colour, sublabel=sub, text_size=19)
        if i < len(causal_steps)-1:
            c.arrow(cx+250, yy+94, cx+250, yy+124, colour=colour, width=2.5)

    # Bottom semantics band
    c.rect(90, 972, 1620, 48, fill=theme.panel_alt, stroke=theme.border, radius=14)
    c.text("observational", 275, 997, size=15, colour=theme.stable, weight=700, anchor="middle", baseline="middle")
    c.text("target-specific observational", 900, 997, size=15, colour=theme.uncertainty, weight=700, anchor="middle", baseline="middle")
    c.text("intervention evidence", 1460, 997, size=15, colour=theme.oppose, weight=700, anchor="middle", baseline="middle")
    c.line(565, 979, 565, 1013, stroke=theme.border, width=1.2)
    c.line(1190, 979, 1190, 1013, stroke=theme.border, width=1.2)

    return c


def evidence_reconstruction(theme: Theme) -> Canvas:
    c = Canvas(1800, 1050, theme)

    # Input and model depth
    c.box_label(55, 410, 210, 130, ["input x", "target y"], accent=theme.selected, sublabel="one sample or batch", text_size=24)

    layer_x = [360, 570, 780, 990, 1200]
    layer_labels = ["h₁", "h₂", "h₃", "h₄", "hᴸ"]
    for i, (x, lab) in enumerate(zip(layer_x, layer_labels, strict=False)):
        colour = mix(theme.support, theme.latent, i/(len(layer_x)-1))
        c.layer_stack(x, 390, 125, 115, 4, colour=colour, label=lab)
        if i == 0:
            c.arrow(275, 475, x-15, 447, colour=theme.selected, width=3.0)
        else:
            c.arrow(layer_x[i-1]+165, 447, x-15, 447, colour=colour, width=3.0)
    c.arrow(1360, 447, 1510, 447, colour=theme.selected, width=3.2)
    c.box_label(1530, 390, 210, 115, ["score sᵧ", "prediction"], accent=theme.selected, sublabel="logit or scalar readout", text_size=22)

    # Metric ribbons above layers
    metric_rows = [
        ("activation energy Eₗ", 120, theme.support),
        ("signed interaction Qₗ", 200, theme.oppose),
        ("predictive uncertainty H(p)", 280, theme.uncertainty),
    ]
    for label, y, colour in metric_rows:
        c.text(label, 60, y+18, size=16, colour=colour, weight=700)
        c.line(270, y+10, 1420, y+10, stroke=theme.grid, width=1.0)
        vals = [0.22, 0.43, 0.66, 0.58, 0.91] if "energy" in label else ([-0.12, 0.18, 0.34, 0.09, 0.52] if "interaction" in label else [0.71, 0.58, 0.41, 0.37, 0.19])
        for x, v in zip(layer_x, vals, strict=False):
            baseline = y+10
            if "interaction" in label:
                top = baseline - v*55
                c.line(x+62, baseline, x+62, top, stroke=colour, width=7, opacity=0.75)
                c.circle(x+62, top, 7, fill=colour, stroke=theme.canvas, stroke_width=1.5)
            else:
                top = baseline - v*50
                c.line(x+62, baseline, x+62, top, stroke=colour, width=7, opacity=0.75)
                c.circle(x+62, top, 7, fill=colour, stroke=theme.canvas, stroke_width=1.5)
        c.text("layer depth", 1435, y+15, size=12.5, colour=theme.text_secondary)

    # Formula panel
    c.rect(100, 620, 650, 350, fill=theme.panel_alt, stroke=theme.border, radius=20)
    c.text("Eₗ = nₗ⁻¹ Σᵢ aₗᵢ²", 145, 685, size=27, colour=theme.support, weight=700)
    c.text("Qₗ = nₗ⁻¹ Σᵢ aₗᵢ · ∂sᵧ/∂aₗᵢ", 145, 755, size=27, colour=theme.oppose, weight=700)
    c.text("H(p) = −Σ p(c) log p(c) / log(C)", 145, 825, size=25, colour=theme.uncertainty, weight=700)
    c.separator(135, 858, 715, 858)
    c.text("robust scaling is applied before cross-layer comparison", 145, 900, size=16, colour=theme.text_secondary)
    c.text("sign, magnitude and uncertainty remain separate channels", 145, 932, size=16, colour=theme.text_secondary)

    # Reasoning pathway reconstruction on right
    c.rect(825, 620, 915, 350, fill=theme.panel, stroke=theme.border, radius=20)
    node_cols = [900, 1110, 1320, 1535]
    node_y = [700, 790, 880]
    # connections first
    edges = [
        (0,0,1,0,0.8,theme.support), (0,0,1,1,0.3,theme.latent),
        (0,1,1,0,0.4,theme.oppose), (0,1,1,2,0.75,theme.oppose),
        (0,2,1,1,0.65,theme.support), (0,2,1,2,0.25,theme.neutral),
        (1,0,2,0,0.85,theme.support), (1,0,2,1,0.35,theme.latent),
        (1,1,2,1,0.62,theme.support), (1,1,2,2,0.22,theme.neutral),
        (1,2,2,1,0.28,theme.oppose), (1,2,2,2,0.80,theme.oppose),
        (2,0,3,0,0.78,theme.support), (2,1,3,0,0.45,theme.support), (2,1,3,1,0.30,theme.uncertainty), (2,2,3,1,0.72,theme.oppose),
    ]
    for c1,r1,c2,r2,val,col in edges:
        c.bezier(node_cols[c1]+22, node_y[r1], node_cols[c2]-22, node_y[r2], colour=col, width=1.0+5.0*val, opacity=0.28+0.55*val)
    labels = [
        ["input group A", "input group B", "input group C"],
        ["community 1", "community 2", "community 3"],
        ["support path", "mixed path", "opposing path"],
        ["selected y", "alternative"],
    ]
    for ci, x in enumerate(node_cols):
        ys = node_y if ci < 3 else [745, 845]
        for ri, y in enumerate(ys):
            col = theme.support if (ci==3 and ri==0) else (theme.oppose if ci==3 else (theme.latent if ri==1 else theme.neutral))
            c.circle(x, y, 19, fill=alpha(theme, col, 0.15), stroke=col, stroke_width=2)
            c.text(labels[ci][ri], x, y+38, size=12.5, colour=theme.text_secondary, anchor="middle")
    c.text("edge width ∝ defined contribution magnitude", 1280, 940, size=14.5, colour=theme.text_secondary, anchor="middle")

    return c


def reasoning_dna_pipeline(theme: Theme) -> Canvas:
    c = Canvas(1800, 1020, theme)

    # Layer-metric matrix
    c.rect(70, 110, 420, 590, fill=theme.panel, stroke=theme.border, radius=20)
    c.text("layers", 115, 160, size=16, colour=theme.text_secondary, weight=700)
    metric_labels = ["E", "σ²", "S", "Q", "H", "t"]
    for j, lab in enumerate(metric_labels):
        c.text(lab, 224+j*38, 180, size=13.5, colour=theme.text_secondary, anchor="middle", weight=700)
    rows, cols = 12, len(metric_labels)
    vals = []
    for r in range(rows):
        row = []
        for col in range(cols):
            v = math.sin((r+1)*(col+2)*0.53)*0.58 + math.cos((r+2)*(col+1)*0.21)*0.35
            row.append(max(-1.0,min(1.0,v)))
        vals.append(row)
    c.matrix(205, 205, rows, cols, 38, vals, negative=theme.oppose, positive=theme.support, neutral=theme.panel_alt)
    for r in range(rows):
        c.text(f"L{r+1}", 170, 205+r*38+19, size=12.5, colour=theme.text_secondary, anchor="end", baseline="middle")
    c.text("M_x ∈ ℝ^(L×M)", 280, 690, size=24, colour=theme.latent, weight=750, anchor="middle")
    c.text("E energy • S sparsity • Q signed interaction • H entropy", 280, 725, size=12.5, colour=theme.text_secondary, anchor="middle")

    # Steps
    step_x = [570, 820, 1065, 1305]
    steps = [
        ("robust scale", "median and IQR", theme.stable),
        ("clip", "bounded influence", theme.uncertainty),
        ("flatten", "fixed ordering", theme.latent),
        ("resample", "dimension d", theme.selected),
    ]
    for i, (lab, sub, col) in enumerate(steps):
        c.box_label(step_x[i], 270, 190, 120, lab, accent=col, sublabel=sub, text_size=19)
        if i == 0:
            c.arrow(500, 330, step_x[i]-12, 330, colour=theme.selected, width=3.0)
        else:
            c.arrow(step_x[i-1]+202, 330, step_x[i]-12, 330, colour=col, width=3.0)

    # Deterministic barcode
    dna_x, dna_y, dna_w, dna_h = 580, 500, 900, 150
    c.rect(dna_x-25, dna_y-35, dna_w+50, dna_h+85, fill=theme.panel_alt, stroke=theme.border, radius=20)
    segments = 64
    for i in range(segments):
        v = math.sin(i*0.91)*0.55 + math.cos(i*0.27)*0.42
        col = theme.support if v>=0 else theme.oppose
        h = 35 + abs(v)*85
        x = dna_x + i*(dna_w/segments)
        c.rect(x, dna_y+dna_h-h, max(2.5,dna_w/segments-2.2), h, fill=mix(theme.panel_alt,col,0.35+0.55*abs(v)), stroke="none", stroke_width=0, radius=2)
    c.text("R(x) = resample(vec(clip(robust_scale(M_x))), d)", dna_x+dna_w/2, dna_y+dna_h+42, size=20, colour=theme.latent, weight=700, anchor="middle")

    # Applications
    apps = [
        ("similarity", "cosine distance", theme.support),
        ("outliers", "reasoning anomalies", theme.oppose),
        ("drift", "distribution change", theme.uncertainty),
        ("comparison", "models and checkpoints", theme.stable),
    ]
    for i, (lab, sub, col) in enumerate(apps):
        x = 160+i*410
        c.box_label(x, 820, 320, 110, lab, accent=col, sublabel=sub, text_size=20)
        c.arrow(dna_x+dna_w/2, dna_y+dna_h+80, x+160, 808, colour=col, width=2.0, bend=35*(i-1.5))

    return c


def latent_community_discovery(theme: Theme) -> Canvas:
    c = Canvas(1800, 1050, theme)

    # Activation dataset
    c.rect(60, 100, 350, 780, fill=theme.panel, stroke=theme.border, radius=22)
    c.text("samples", 100, 155, size=17, colour=theme.text_secondary, weight=700)
    # grid of activation profiles
    rows, cols = 18, 12
    vals = []
    for r in range(rows):
        vals.append([math.sin((r+1)*0.37+(cc+1)*0.62)*0.8 for cc in range(cols)])
    c.matrix(100, 190, rows, cols, 22, vals, negative=theme.oppose, positive=theme.support, neutral=theme.panel_alt)
    c.text("A ∈ ℝᴺˣᵁ", 235, 650, size=28, colour=theme.support, weight=750, anchor="middle")
    c.text("unit activation profiles", 235, 690, size=16, colour=theme.text_secondary, anchor="middle")
    c.text("standardise within layer", 235, 770, size=15, colour=theme.text_secondary, anchor="middle")
    c.text("retain sample correspondence", 235, 802, size=15, colour=theme.text_secondary, anchor="middle")

    # Similarity matrix
    c.arrow(430, 490, 555, 490, colour=theme.selected, width=3.0)
    c.rect(580, 130, 420, 650, fill=theme.panel_alt, stroke=theme.border, radius=22)
    c.text("unit similarity", 625, 180, size=17, colour=theme.text_secondary, weight=700)
    sim = []
    for r in range(12):
        row=[]
        for cc in range(12):
            if r//4 == cc//4:
                v=0.72+0.25*math.cos((r-cc)*0.8)
            else:
                v=0.05+0.15*math.sin((r+1)*(cc+2))
            row.append(max(-1,min(1,v)))
        sim.append(row)
    c.matrix(655, 220, 12, 12, 24, sim, negative=theme.oppose, positive=theme.latent, neutral=theme.panel_alt)
    c.text("Cᵢⱼ = corr(A·ᵢ, A·ⱼ)", 790, 560, size=24, colour=theme.latent, weight=700, anchor="middle")
    c.pill(650, 630, 280, 42, "threshold or weighted graph", colour=theme.latent)
    c.text("method and seed stored in metadata", 790, 724, size=14.5, colour=theme.text_secondary, anchor="middle")

    # Community graph
    c.arrow(1020, 490, 1130, 490, colour=theme.selected, width=3.0)
    c.rect(1150, 90, 590, 600, fill=theme.panel, stroke=theme.border, radius=22)
    centres = [(1280,250),(1470,210),(1605,320),(1370,430),(1570,500)]
    cols_c = [theme.support,theme.latent,theme.stable,theme.oppose,theme.uncertainty]
    # inter edges
    for a,b,w,col in [(0,1,3,theme.selected),(1,2,2,theme.latent),(0,3,2,theme.support),(3,4,3,theme.oppose),(2,4,1.5,theme.uncertainty),(1,3,1.2,theme.neutral)]:
        c.bezier(centres[a][0],centres[a][1],centres[b][0],centres[b][1],colour=col,width=w,opacity=0.55)
    for idx,(cx,cy) in enumerate(centres):
        col=cols_c[idx]
        # satellites
        for k in range(6):
            ang=2*math.pi*k/6+idx*0.3
            sx=cx+48*math.cos(ang)
            sy=cy+48*math.sin(ang)
            c.line(cx,cy,sx,sy,stroke=alpha(theme,col,0.65),width=1.4)
            c.circle(sx,sy,7,fill=alpha(theme,col,0.35),stroke=col,stroke_width=1.3)
        c.circle(cx,cy,23,fill=alpha(theme,col,0.16),stroke=col,stroke_width=2.2)
        c.text(f"C{idx+1}",cx,cy+5,size=14,weight=750,anchor="middle")
    c.text("community detection", 1445, 610, size=17, colour=theme.text_secondary, weight=700, anchor="middle")

    # Concept candidates and validation
    c.arrow(1445, 705, 1445, 765, colour=theme.selected, width=3.0)
    c.rect(570, 855, 1170, 140, fill=theme.panel_alt, stroke=theme.border, radius=20)
    labels=[
        ("latent group", "unlabelled structure", theme.latent),
        ("concept candidate", "external examples or probes", theme.support),
        ("intervention", "ablate community", theme.oppose),
        ("supported concept", "only after validation", theme.stable),
    ]
    for i,(lab,sub,col) in enumerate(labels):
        x=600+i*280
        c.box_label(x,880,240,86,lab,accent=col,sublabel=sub,text_size=16)
        if i<len(labels)-1:
            c.arrow(x+250,923,x+270,923,colour=col,width=2.0)

    return c


def training_monitoring(theme: Theme) -> Canvas:
    c = Canvas(1800, 1020, theme)

    # Training loop ring
    cx, cy = 430, 500
    nodes = [
        (cx, 160, "batch", theme.support),
        (690, 300, "forward", theme.support_strong),
        (690, 620, "loss + backward", theme.uncertainty),
        (cx, 780, "optimiser step", theme.stable),
        (170, 620, "checkpoint", theme.latent),
        (170, 300, "next epoch", theme.selected),
    ]
    for i,(x,y,label,col) in enumerate(nodes):
        c.circle(x,y,62,fill=alpha(theme,col,0.10),stroke=col,stroke_width=2.5)
        c.text(label,x,y+5,size=17,weight=700,anchor="middle",baseline="middle")
        x2,y2,_,col2=nodes[(i+1)%len(nodes)]
        c.arrow(x+0.68*(x2-x)/max(1,math.hypot(x2-x,y2-y))*62, y+0.68*(y2-y)/max(1,math.hypot(x2-x,y2-y))*62,
                x2-0.68*(x2-x)/max(1,math.hypot(x2-x,y2-y))*62, y2-0.68*(y2-y)/max(1,math.hypot(x2-x,y2-y))*62,
                colour=col2,width=2.4,bend=18)
    c.text("scoped VRAVEN hooks", cx, cy-18, size=25, weight=800, anchor="middle")
    c.text("capture at selected intervals", cx, cy+22, size=16, colour=theme.text_secondary, anchor="middle")

    # Epoch trace timeline
    c.rect(840, 90, 870, 245, fill=theme.panel, stroke=theme.border, radius=22)
    c.text("epoch trace", 890, 142, size=18, colour=theme.text_secondary, weight=700)
    epochs=[1,5,10,20,40,80]
    for i,e in enumerate(epochs):
        x=930+i*132
        col=mix(theme.support,theme.stable,i/(len(epochs)-1))
        c.circle(x,225,18,fill=alpha(theme,col,0.16),stroke=col,stroke_width=2)
        c.text(str(e),x,230,size=14,weight=700,anchor="middle")
        if i<len(epochs)-1:
            c.arrow(x+25,225,x+107,225,colour=theme.text_secondary,width=1.8)
        c.text(f"Rₑ{e}",x,285,size=14,colour=col,anchor="middle",weight=650)
    c.arrow(745,620,820,230,colour=theme.latent,width=2.5,bend=-60)

    # Diagnostics panels
    diagnostics=[
        ("gradient flow", "vanishing • exploding", theme.uncertainty),
        ("activation health", "dead • saturated units", theme.oppose),
        ("class geometry", "separation and overlap", theme.support),
        ("reasoning stability", "fingerprint convergence", theme.stable),
        ("concept formation", "community persistence", theme.latent),
        ("drift across epochs", "JSD / MMD / trajectory", theme.selected),
    ]
    for i,(lab,sub,col) in enumerate(diagnostics):
        row=i//2
        cc=i%2
        x=855+cc*430
        y=395+row*175
        c.box_label(x,y,380,115,lab,accent=col,sublabel=sub,text_size=19)
        c.arrow(1275,335,x+190,y-12,colour=col,width=1.7,bend=25*(cc*2-1))

    # Comparison branch
    c.rect(840, 925, 870, 65, fill=theme.panel_alt, stroke=theme.border, radius=16)
    c.text("after training", 900, 960, size=15, colour=theme.latent, weight=700, baseline="middle")
    c.text("load any selected checkpoint and reproduce the same trace path", 1035, 960, size=15, colour=theme.text_secondary, baseline="middle")

    return c


def architecture_adapters(theme: Theme) -> Canvas:
    c = Canvas(1800, 1050, theme)

    # Architectures on left/top
    architectures = [
        ("MLP", "features → hidden units", theme.support, False),
        ("CNN / ResNet", "spatial channels and blocks", theme.support_strong, False),
        ("Transformer", "tokens • attention • residuals", theme.latent, False),
        ("Autoencoder", "encoder • latent • decoder", theme.stable, False),
        ("GAN", "generator / discriminator", theme.uncertainty, True),
        ("GNN", "nodes • edges • messages", theme.neutral, True),
        ("LLM", "tokens • heads • residual stream", theme.neutral, True),
    ]
    for i,(label,sub,col,dashed) in enumerate(architectures):
        row=i//4
        cc=i%4
        x=65+cc*425
        y=70+row*175
        c.box_label(x,y,360,115,label,accent=col,sublabel=sub,text_size=20,dashed=dashed)
        if dashed:
            c.text("specialised adapter planned",x+180,y+142,size=12.5,colour=theme.text_secondary,anchor="middle")

    # Unified semantic mapping
    c.rect(160, 490, 1480, 205, fill=theme.panel_alt, stroke=theme.border, radius=26)
    c.text("unified capture semantics", 900, 535, size=24, weight=800, anchor="middle")
    semantics=[
        ("ordered module", theme.support),
        ("activation summary", theme.support_strong),
        ("target interaction", theme.oppose),
        ("representation profile", theme.latent),
        ("uncertainty source", theme.uncertainty),
        ("intervention point", theme.stable),
    ]
    for i,(lab,col) in enumerate(semantics):
        x=220+i*225
        c.pill(x,595,190,46,lab,colour=col,text_size=14.5)
    # arrows from architectures
    for i in range(len(architectures)):
        row=i//4
        cc=i%4
        x=65+cc*425+180
        y=70+row*175+115
        c.arrow(x,y+5,900,480,colour=architectures[i][2],width=1.6,dash="7 6" if architectures[i][3] else None,bend=20*(cc-1.5))

    # Unified trace then architecture-specific views
    c.arrow(900,707,900,765,colour=theme.selected,width=3.2)
    c.box_label(730,780,340,105,["NeuralTrace", "common schema"],accent=theme.selected,sublabel="portable analyses",text_size=22)

    views=[
        ("spatial map", theme.support),
        ("token flow", theme.latent),
        ("latent path", theme.stable),
        ("message graph", theme.uncertainty),
        ("generic DNA", theme.selected),
    ]
    for i,(lab,col) in enumerate(views):
        x=160+i*300
        c.box_label(x,930,250,80,lab,accent=col,text_size=17)
        c.arrow(900,895,x+125,918,colour=col,width=1.8,bend=25*(i-2))

    return c
