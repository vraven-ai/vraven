from __future__ import annotations

import math

from .primitives import Canvas, alpha, regular_polygon
from .theme import Theme


def benchmark_protocol(theme: Theme) -> Canvas:
    c = Canvas(1800, 1050, theme)

    # Controlled inputs
    c.rect(55, 110, 310, 780, fill=theme.panel, stroke=theme.border, radius=24)
    c.box_label(90, 155, 240, 95, ["fixed model", "checkpoint"], accent=theme.selected, sublabel="same weights and mode", text_size=20)
    c.box_label(90, 285, 240, 95, ["fixed inputs", "and targets"], accent=theme.support, sublabel="same order and seed", text_size=20)
    c.box_label(90, 415, 240, 95, ["fixed hardware", "and precision"], accent=theme.stable, sublabel="warm-up disclosed", text_size=20)
    c.box_label(90, 545, 240, 95, ["method budget", "matched where possible"], accent=theme.uncertainty, sublabel="calls • samples • steps", text_size=18)
    c.box_label(90, 675, 240, 95, ["task-specific", "evaluation"], accent=theme.latent, sublabel="vision • text • tabular", text_size=19)
    c.text("controlled benchmark object", 210, 835, size=15, colour=theme.text_secondary, anchor="middle", weight=700)

    # Methods fan-out
    methods = [
        ("VRAVEN", theme.selected),
        ("SHAP", theme.support),
        ("LIME", theme.stable),
        ("Integrated Gradients", theme.latent),
        ("DeepLIFT", theme.uncertainty),
        ("Grad-CAM / task baseline", theme.neutral),
    ]
    c.arrow(380, 500, 500, 500, colour=theme.selected, width=3.0)
    c.rect(520, 90, 470, 830, fill=theme.panel_alt, stroke=theme.border, radius=24)
    for i,(label,col) in enumerate(methods):
        y=145+i*120
        c.box_label(570,y,370,82,label,accent=col,sublabel="same prediction target",text_size=18)
        c.arrow(500,500,558,y+41,colour=col,width=1.7,bend=(y+41-500)*0.18)

    # Evaluation metrics in matrix layout
    c.arrow(1005, 500, 1125, 500, colour=theme.selected, width=3.0)
    c.rect(1145, 90, 600, 830, fill=theme.panel, stroke=theme.border, radius=24)
    metric_groups = [
        ("efficiency", ["wall time", "model calls", "peak memory", "energy if available"], theme.stable),
        ("fidelity", ["deletion", "insertion", "intervention effect", "local surrogate fit"], theme.oppose),
        ("robustness", ["stability", "sensitivity", "rank consistency", "seed reproducibility"], theme.uncertainty),
        ("coverage", ["architecture support", "device support", "output support", "failure modes"], theme.support),
        ("interpretation", ["human study", "task utility", "calibration", "scientific clarity"], theme.latent),
    ]
    for i,(group,items,col) in enumerate(metric_groups):
        y=130+i*145
        c.rect(1185,y,520,118,fill=alpha(theme,col,0.045),stroke=alpha(theme,col,0.55),stroke_width=1.4,radius=14)
        c.text(group,1215,y+32,size=18,colour=col,weight=750)
        for j,item in enumerate(items):
            xx=1215+(j%2)*240
            yy=y+66+(j//2)*28
            c.circle(xx,yy,5,fill=col,stroke=col,stroke_width=0)
            c.text(item,xx+14,yy+1,size=13.5,colour=theme.text_secondary,baseline="middle")

    # Statistical reporting footer
    c.rect(55, 945, 1690, 60, fill=theme.panel_alt, stroke=theme.border, radius=16)
    c.text("report distributions, confidence intervals and paired tests; avoid single-run headline numbers", 900, 976, size=17, colour=theme.text_secondary, anchor="middle", baseline="middle", weight=650)

    return c


def causal_validation_loop(theme: Theme) -> Canvas:
    c = Canvas(1700, 1050, theme)

    cx, cy = 850, 500
    steps = [
        (850, 125, "candidate pathway P", "ranked observationally", theme.uncertainty),
        (1265, 295, "intervention I(P)", "zero • scale • replace", theme.oppose),
        (1265, 705, "re-inference", "same input and state", theme.support),
        (850, 890, "effect estimate Δsᵧ", "original minus intervened", theme.oppose_strong),
        (435, 705, "matched controls", "random / magnitude matched", theme.stable),
        (435, 295, "repeat and aggregate", "samples • seeds • layers", theme.latent),
    ]
    for i,(x,y,label,sub,col) in enumerate(steps):
        c.box_label(x-155,y-60,310,120,label,accent=col,sublabel=sub,text_size=19)
        nx,ny,_,_,ncol=steps[(i+1)%len(steps)]
        # curved clockwise arrows
        angle = math.atan2(ny-y,nx-x)
        sx=x+155*math.cos(angle)
        sy=y+60*math.sin(angle)
        ex=nx-155*math.cos(angle)
        ey=ny-60*math.sin(angle)
        c.arrow(sx,sy,ex,ey,colour=ncol,width=2.8,bend=38)

    # Centre equation and safeguards
    c.polygon(regular_polygon(cx,cy,185,6),fill=alpha(theme,theme.selected,0.07),stroke=theme.selected,width=2.2)
    c.text("Δsᵧ(P) = sᵧ(x) − sᵧ(x; I(P))",cx,cy-25,size=26,colour=theme.selected,weight=800,anchor="middle")
    c.text("causal support for the defined intervention",cx,cy+18,size=16,colour=theme.text_secondary,anchor="middle")
    c.separator(cx-135,cy+48,cx+135,cy+48)
    c.text("not recovery of the model's true mechanism",cx,cy+85,size=14.5,colour=theme.oppose,anchor="middle",weight=650)

    # Bottom inference standards
    c.rect(130, 975, 1440, 52, fill=theme.panel_alt, stroke=theme.border, radius=14)
    standards=[("state restored",theme.stable),("hook removed",theme.support),("effect signed",theme.oppose),("controls disclosed",theme.latent),("uncertainty reported",theme.uncertainty)]
    for i,(label,col) in enumerate(standards):
        x=235+i*285
        c.circle(x,1001,6,fill=col,stroke=col,stroke_width=0)
        c.text(label,x+14,1002,size=14.5,colour=theme.text_secondary,baseline="middle")

    return c


def reliability_calibration(theme: Theme) -> Canvas:
    c = Canvas(1800, 1030, theme)

    # Separate evidence channels
    channels=[
        ("prediction confidence", "calibrated output probability", theme.support),
        ("explanation stability", "R(x) under valid perturbations", theme.stable),
        ("method agreement", "rank • sign • top-k overlap", theme.latent),
        ("causal support", "Δsᵧ under controlled intervention", theme.oppose),
        ("distribution support", "distance from reference traces", theme.uncertainty),
    ]
    for i,(lab,sub,col) in enumerate(channels):
        y=95+i*165
        c.box_label(70,y,450,115,lab,accent=col,sublabel=sub,text_size=20)
        c.arrow(535,y+57,720,500,colour=col,width=2.2,bend=(y+57-500)*0.15)

    # Fusion core, but not a single arbitrary score
    c.polygon(regular_polygon(900,500,190,8),fill=alpha(theme,theme.selected,0.07),stroke=theme.selected,width=2.4)
    c.text("reliability profile",900,455,size=28,weight=800,anchor="middle")
    c.text("multidimensional",900,493,size=19,colour=theme.selected,weight=700,anchor="middle")
    c.text("retain each source",900,535,size=15.5,colour=theme.text_secondary,anchor="middle")
    c.text("calibrate before aggregation",900,565,size=15.5,colour=theme.text_secondary,anchor="middle")

    # Output profile on right
    c.rect(1180,100,550,785,fill=theme.panel,stroke=theme.border,radius=24)
    axes=[
        ("confidence",0.82,theme.support),
        ("stability",0.67,theme.stable),
        ("agreement",0.59,theme.latent),
        ("causal support",0.48,theme.oppose),
        ("distribution support",0.74,theme.uncertainty),
    ]
    for i,(lab,val,col) in enumerate(axes):
        y=175+i*120
        c.text(lab,1225,y,size=16,colour=theme.text_secondary,weight=650)
        c.rect(1225,y+25,430,22,fill=theme.panel_alt,stroke=theme.border,stroke_width=1.0,radius=11)
        c.rect(1225,y+25,430*val,22,fill=alpha(theme,col,0.80),stroke=col,stroke_width=0,radius=11)
        c.text("illustrative",1655,y+42,size=11.5,colour=theme.text_secondary,anchor="end")
    c.separator(1225,790,1685,790)
    c.text("do not label a composite as confidence",1455,830,size=15.5,colour=theme.oppose,weight=700,anchor="middle")
    c.text("unless it is empirically calibrated and validated",1455,858,size=14.5,colour=theme.text_secondary,anchor="middle")

    # Calibrated composite path below centre
    c.box_label(690,785,420,105,["optional calibrated model", "g(z) → probability of explanation success"],accent=theme.selected,sublabel="trained on defined validation outcomes",text_size=18,dashed=True)
    c.arrow(900,695,900,772,colour=theme.selected,width=2.5,dash="7 6")

    return c


def reasoning_drift(theme: Theme) -> Canvas:
    c = Canvas(1800, 1040, theme)

    # Two domains/time windows
    for idx,(x,label,col) in enumerate([(60,"reference window t₀",theme.support),(60,"current window t₁",theme.oppose)]):
        y=120+idx*390
        c.box_label(x,y,290,100,label,accent=col,sublabel="same model and trace configuration",text_size=19)
        # sample traces
        for r in range(4):
            for cc in range(4):
                px=x+45+cc*55
                py=y+155+r*48
                jitter=math.sin((r+1)*(cc+2)+idx*0.8)
                c.circle(px+7*jitter,py+5*math.cos(cc+idx),8,fill=alpha(theme,col,0.22),stroke=col,stroke_width=1.2)
        c.text("independent traces R(xᵢ)",x+145,y+360,size=14.5,colour=theme.text_secondary,anchor="middle")

    # Fingerprint distributions
    c.arrow(370,315,520,315,colour=theme.support,width=2.5)
    c.arrow(370,705,520,705,colour=theme.oppose,width=2.5)
    c.rect(545,90,520,820,fill=theme.panel_alt,stroke=theme.border,radius=24)
    c.text("shared fingerprint space",805,145,size=20,weight=750,anchor="middle")
    # scatter with shared coordinates
    for i in range(60):
        a=2*math.pi*i/60
        rr=110+35*math.sin(i*1.7)
        x=760+rr*math.cos(a)*0.85
        y=365+rr*math.sin(a)*0.55
        c.circle(x,y,5.5,fill=alpha(theme,theme.support,0.35),stroke=theme.support,stroke_width=0.8)
    for i in range(60):
        a=2*math.pi*i/60
        rr=105+38*math.cos(i*1.35)
        x=845+rr*math.cos(a)*0.82
        y=615+rr*math.sin(a)*0.50
        c.circle(x,y,5.5,fill=alpha(theme,theme.oppose,0.35),stroke=theme.oppose,stroke_width=0.8)
    c.pill(650,810,310,42,"projection fitted once or aligned",colour=theme.latent,text_size=14.5)
    c.text("sample correspondence and seed retained",805,872,size=14,colour=theme.text_secondary,anchor="middle")

    # Statistical decisions
    c.arrow(1080,500,1200,500,colour=theme.selected,width=3.0)
    c.rect(1220,90,520,820,fill=theme.panel,stroke=theme.border,radius=24)
    tests=[
        ("distribution distance", "Jensen–Shannon or MMD", theme.latent),
        ("location and scale", "energy distance / Wasserstein", theme.support),
        ("permutation test", "null from label exchange", theme.stable),
        ("class-conditional drift", "compare within predictions", theme.uncertainty),
        ("effect localisation", "layers and communities", theme.oppose),
    ]
    for i,(lab,sub,col) in enumerate(tests):
        y=135+i*145
        c.box_label(1260,y,440,105,lab,accent=col,sublabel=sub,text_size=18)
    c.rect(1220,940,520,65,fill=alpha(theme,theme.oppose,0.04),stroke=alpha(theme,theme.oppose,0.55),radius=14)
    c.text("reasoning drift may occur before accuracy changes",1480,975,size=15.5,colour=theme.oppose,weight=700,anchor="middle",baseline="middle")

    return c


def reproducibility_provenance(theme: Theme) -> Canvas:
    c = Canvas(1800, 1030, theme)

    # Source artefacts
    sources=[
        ("model", "structure hash • parameter count", theme.support),
        ("input", "shape • dtype • bounded fingerprint", theme.stable),
        ("environment", "Python • framework • CUDA/MPS", theme.latent),
        ("configuration", "mode • target • capture policy", theme.uncertainty),
        ("randomness", "seed • deterministic flags", theme.oppose),
    ]
    for i,(lab,sub,col) in enumerate(sources):
        y=90+i*170
        c.box_label(70,y,410,115,lab,accent=col,sublabel=sub,text_size=21)
        c.arrow(495,y+58,720,510,colour=col,width=2.0,bend=(y+58-510)*0.14)

    # Provenance bundle
    c.rect(735,130,420,760,fill=theme.panel_alt,stroke=theme.selected,stroke_width=2.3,radius=28)
    c.text("reproducibility bundle",945,185,size=26,weight=800,anchor="middle")
    c.pill(805,230,280,42,"NeuralTrace schema",colour=theme.selected)
    records=[
        ("trace identifier",theme.support),
        ("timestamp and version",theme.stable),
        ("ordered layer records",theme.latent),
        ("prediction and target",theme.uncertainty),
        ("capture metadata",theme.support),
        ("intervention records",theme.oppose),
        ("visual contract",theme.selected),
        ("file hashes",theme.stable),
    ]
    for i,(lab,col) in enumerate(records):
        y=315+i*61
        c.circle(795,y,7,fill=col,stroke=col,stroke_width=0)
        c.text(lab,818,y+1,size=16,colour=theme.text_secondary,baseline="middle")
        c.line(1030,y,1095,y,stroke=alpha(theme,col,0.5),width=3)
    c.text("serialise compact data, not arbitrary Python objects",945,842,size=14.5,colour=theme.text_secondary,anchor="middle")

    # Outputs and verification
    c.arrow(1170,510,1305,510,colour=theme.selected,width=3.0)
    outputs=[
        ("trace.json", "machine-readable record", theme.selected),
        ("figure metadata", "metric version and mapping", theme.support),
        ("release manifest", "SHA-256 inventory", theme.stable),
        ("environment lock", "dependencies and platform", theme.latent),
        ("reproduction script", "deterministic regeneration", theme.uncertainty),
    ]
    for i,(lab,sub,col) in enumerate(outputs):
        y=90+i*170
        c.box_label(1320,y,410,115,lab,accent=col,sublabel=sub,text_size=20)

    c.rect(70,950,1660,48,fill=theme.panel_alt,stroke=theme.border,radius=13)
    c.text("a figure is reproducible only when the underlying trace, transformation and rendering configuration are recoverable",900,975,size=15.5,colour=theme.text_secondary,weight=650,anchor="middle",baseline="middle")

    return c
