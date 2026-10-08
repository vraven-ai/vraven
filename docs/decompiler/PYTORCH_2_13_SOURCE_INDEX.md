# PyTorch 2.13 Source Index for VRAVEN

VRAVEN does not mirror the entire PyTorch documentation. It records the parts that define the execution and transformation contracts required by the neural decompiler.

| PyTorch area | VRAVEN use | Official documentation |
| --- | --- | --- |
| `torch.nn.Module` | Parameters, buffers, state and fallback instrumentation | https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html |
| `torch.export` | Preferred normalised, ahead-of-time graph capture | https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/export.html |
| Export IR | Graph signatures, ATen graph form and constraints | https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/export/ir_spec.html |
| FX | Graph interpreter, graph rewriting and generated execution | https://docs.pytorch.org/docs/2.13/fx.html |
| Autograd | Reverse differentiation and local derivative screening | https://docs.pytorch.org/docs/2.13/autograd.html |
| `torch.func` | Future batched JVP, VJP, Jacobian and intervention evaluation | https://docs.pytorch.org/docs/2.13/func.html |
| `functional_call` | Functional model execution under supplied state | https://docs.pytorch.org/docs/2.13/generated/torch.func.functional_call.html |
| `functionalize` | Mutation and view removal before transformation | https://docs.pytorch.org/docs/2.13/generated/torch.func.functionalize.html |
| Profiler | Operator time, memory, shape and source evidence | https://docs.pytorch.org/docs/2.13/profiler.html |
| Custom operators | Abstract behaviour, fake tensors, autograd and vectorisation contracts | https://docs.pytorch.org/docs/2.13/library.html |
| Compiler provenance | Planned concept-to-ATen-to-generated-kernel trace | https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_inductor_provenance.html |
| Meta tensors | Planned allocation-free structural reconnaissance | https://docs.pytorch.org/docs/2.13/meta.html |

## Stability policy

Public PyTorch APIs are preferred. Private or unstable compiler APIs must remain behind version-specific adapters and must never become the only path to a valid VRAVEN result.

## Knowledge boundary

PyTorch documentation explains how computation is represented and executed. It does not establish the semantic meaning of a learned representation. VRAVEN therefore keeps computation capture, concept discovery, causal testing and semantic grounding as separate evidence layers.
