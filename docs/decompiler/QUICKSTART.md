# Quick Start

## Install from the repository

```bash
python -m pip install -e ".[analysis,publication]"
vraven doctor
```

## Python API

```python
import torch
import vraven

model = MyModel().eval()
reference = torch.load("reference.pt", weights_only=True)

result = vraven.reverse(
    model,
    (reference,),
    focus_args=(reference[:1],),
    target=1,
    class_names={0: "Benign", 1: "Malware"},
    config=vraven.DecompileConfig.quick(),
)

print(result.summary())
result.save("vraven_decision_proof")
result.write_report("vraven_decision_proof.html", format="html")
```

## Command line for a state dictionary

```bash
vraven reverse model.pt \
  --model-factory my_project.models:Classifier \
  --factory-kwargs '{"input_dim":476,"num_classes":2}' \
  --reference reference.npy \
  --labels labels.npy \
  --target 1 \
  --class-names Benign,Malware \
  --profile standard \
  --report-format html \
  --output vraven_decision_proof.html
```

A state dictionary does not contain an arbitrary Python architecture. The model factory is required.

## Capture only

```bash
vraven capture model.pt \
  --model-factory my_project.models:Classifier \
  --factory-kwargs '{"input_dim":476,"num_classes":2}' \
  --input sample.npy \
  --output mechanism_ir.json
```

## Demonstration

```bash
vraven decompile-demo --profile quick --report-format html --output vraven_demo.html
```
