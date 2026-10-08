from __future__ import annotations

from .primitives import Canvas, alpha
from .theme import Theme


def visual_contract(theme: Theme) -> Canvas:
    c = Canvas(1800, 1040, theme)

    # Measured data sources
    c.rect(60, 90, 330, 790, fill=theme.panel, stroke=theme.border, radius=24)
    sources=[
        ("signed contribution", "value and zero centre", theme.support),
        ("uncertainty", "named estimator", theme.uncertainty),
        ("layer order", "real model depth", theme.latent),
        ("community identity", "deterministic mapping", theme.stable),
        ("causal effect", "defined intervention", theme.oppose),
    ]
    for i,(lab,sub,col) in enumerate(sources):
        y=135+i*135
        c.box_label(95,y,260,92,lab,accent=col,sublabel=sub,text_size=17)
    c.text("validated arrays only",225,835,size=15,colour=theme.text_secondary,anchor="middle",weight=700)

    # Encoding contract
    c.arrow(405,485,540,485,colour=theme.selected,width=3.0)
    c.rect(560,90,510,790,fill=theme.panel_alt,stroke=theme.border,radius=24)
    encodings=[
        ("position", "layer • token • community", theme.selected),
        ("sign", "above/below zero or direction", theme.oppose),
        ("magnitude", "width • area • height", theme.support),
        ("colour", "semantic role, never decoration", theme.latent),
        ("opacity", "secondary magnitude only", theme.neutral),
        ("annotation", "metric • units • definition", theme.stable),
    ]
    for i,(lab,sub,col) in enumerate(encodings):
        y=130+i*112
        c.circle(615,y+22,11,fill=alpha(theme,col,0.18),stroke=col,stroke_width=2)
        c.text(lab,645,y+18,size=18,colour=col,weight=750)
        c.text(sub,645,y+46,size=14.5,colour=theme.text_secondary)
    c.separator(605,820,1020,820)
    c.text("VisualContract + metric metadata",815,853,size=16,colour=theme.selected,weight=700,anchor="middle")

    # Standalone figure core
    c.arrow(1085,485,1215,485,colour=theme.selected,width=3.0)
    c.rect(1235,270,500,430,fill=theme.panel,stroke=theme.selected,stroke_width=2.2,radius=26)
    # miniature signed flow figure
    c.text("one scientific question",1485,320,size=19,weight=800,anchor="middle")
    xcols=[1305,1425,1545,1660]
    yvals=[[390,460,545],[380,470,555],[400,455,540],[420,515]]
    for ci in range(3):
        for ri,y in enumerate(yvals[ci]):
            for rj,y2 in enumerate(yvals[ci+1]):
                val=((ri+1)*(rj+2)+ci)%5
                col=theme.support if (ri+rj+ci)%2==0 else theme.oppose
                c.bezier(xcols[ci]+10,y,xcols[ci+1]-10,y2,colour=col,width=1.0+val*0.7,opacity=0.25+val*0.10)
    for ci,ys in enumerate(yvals):
        for ri,y in enumerate(ys):
            col=theme.latent if ri==1 else (theme.support if ri==0 else theme.oppose)
            c.circle(xcols[ci],y,10,fill=alpha(theme,col,0.22),stroke=col,stroke_width=1.6)
    c.text("edge width ∝ measured contribution",1485,650,size=14,colour=theme.text_secondary,anchor="middle")

    # Two modes and outputs
    c.arrow(1485,710,1485,775,colour=theme.selected,width=3.0)
    c.box_label(1070,815,330,115,["interactive_dark", "hover • zoom • filtering"],accent=theme.selected,sublabel="browser and Studio",text_size=18)
    c.box_label(1435,815,330,115,["publication_light", "white • vector • printable"],accent=theme.support,sublabel="paper and thesis",text_size=18)
    c.arrow(1485,775,1235,803,colour=theme.selected,width=2.0,bend=-35)
    c.arrow(1485,775,1600,803,colour=theme.support,width=2.0,bend=35)
    c.text("same data • same geometry • same scientific meaning",1415,982,size=15.5,colour=theme.text_secondary,anchor="middle",weight=650)

    # semantic palette footer
    items=[("support",theme.support),("oppose",theme.oppose),("uncertainty",theme.uncertainty),("stable",theme.stable),("latent",theme.latent),("neutral",theme.neutral)]
    for i,(lab,col) in enumerate(items):
        x=110+i*190
        c.circle(x,970,7,fill=col,stroke=col,stroke_width=0)
        c.text(lab,x+15,971,size=13.5,colour=theme.text_secondary,baseline="middle")

    return c


def user_workflow(theme: Theme) -> Canvas:
    c = Canvas(1800, 1020, theme)

    # Install lane
    c.rect(55, 70, 1690, 120, fill=theme.panel_alt, stroke=theme.border, radius=22)
    c.box_label(90, 95, 400, 70, "local wheel", accent=theme.stable, sublabel="python -m pip install vraven-…whl", text_size=18)
    c.box_label(700, 95, 400, 70, "editable source", accent=theme.latent, sublabel="python -m pip install -e .", text_size=18)
    c.box_label(1310, 95, 400, 70, "PyPI release", accent=theme.neutral, sublabel="future: pip install vraven", text_size=18, dashed=True)
    c.arrow(500,130,685,130,colour=theme.text_secondary,width=1.8)
    c.arrow(1110,130,1295,130,colour=theme.text_secondary,width=1.8,dash="7 6")

    # Two model entry paths
    c.rect(55, 245, 520, 650, fill=theme.panel, stroke=theme.border, radius=24)
    c.text("live model path",315,295,size=22,weight=800,anchor="middle")
    c.box_label(105, 350, 420, 100, ["model: torch.nn.Module", "inputs: Tensor / nested structure"], accent=theme.support, sublabel="richest access", text_size=19)
    c.arrow(315,462,315,520,colour=theme.selected,width=2.8)
    c.box_label(105,535,420,100,["vraven.explain(model, inputs)", "target • mode • class_names"],accent=theme.selected,sublabel="fast • gradient • causal",text_size=18)
    c.arrow(315,647,315,705,colour=theme.selected,width=2.8)
    c.box_label(105,720,420,100,["VRavenReport", "figures • trace • HTML"],accent=theme.latent,sublabel="standalone outputs",text_size=20)

    c.rect(640, 245, 520, 650, fill=theme.panel_alt, stroke=theme.border, radius=24)
    c.text("saved weights path",900,295,size=22,weight=800,anchor="middle")
    c.box_label(690,350,420,100,["weights.pt / checkpoint.pth", "trusted state dictionary"],accent=theme.stable,sublabel="weights do not define architecture",text_size=18)
    c.arrow(900,462,900,520,colour=theme.selected,width=2.8)
    c.box_label(690,535,420,100,["model_factory()", "preprocessing and readout"],accent=theme.uncertainty,sublabel="reconstruct the model safely",text_size=18)
    c.arrow(900,647,900,705,colour=theme.selected,width=2.8)
    c.box_label(690,720,420,100,["load_pytorch_model", "then vraven.explain"],accent=theme.selected,sublabel="same report contract",text_size=19)

    # Output family
    c.rect(1225,245,520,650,fill=theme.panel,stroke=theme.border,radius=24)
    c.text("report outputs",1485,295,size=22,weight=800,anchor="middle")
    outputs=[
        ("decision_flow()",theme.support),
        ("reasoning_dna()",theme.latent),
        ("conflict_field()",theme.uncertainty),
        ("causal_pathway()",theme.oppose),
        ("export_html()",theme.selected),
        ("save_trace_json()",theme.stable),
    ]
    for i,(lab,col) in enumerate(outputs):
        y=350+i*78
        c.box_label(1285,y,400,60,lab,accent=col,text_size=17)
    c.pill(1320,850,330,42,"publication_light or interactive_dark",colour=theme.selected,text_size=13.5)

    c.arrow(585,570,1210,570,colour=theme.selected,width=2.4,bend=-70)
    c.arrow(1170,570,1210,570,colour=theme.selected,width=2.4)

    c.rect(55,940,1690,50,fill=theme.panel_alt,stroke=theme.border,radius=14)
    c.text("unsafe pickled model objects remain blocked by default; only load artefacts you trust",900,966,size=15.5,colour=theme.oppose,weight=700,anchor="middle",baseline="middle")

    return c
