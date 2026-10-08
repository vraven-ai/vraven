# PyTorch 2.13 Technical Basis

VRAVEN is designed around capabilities documented in PyTorch 2.13. Supported
Python versions are exercised in continuous integration, while `vraven doctor`
records the exact local Python, PyTorch, device and export capabilities used for
an individual analysis.

## `torch.export`

VRAVEN prefers `torch.export` because it produces a normalised ahead-of-time graph with graph signatures, parameters, buffers and symbolic shape constraints. The capture adapter optionally runs decompositions to reduce mutation and aliasing complexity.

Official reference:

* https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/export.html
* https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/export/ir_spec.html

## FX

FX supplies the graph representation, interpreter and rewriting system used for recording intermediate values and compiling intervention gates.

Official reference:

* https://docs.pytorch.org/docs/2.13/fx.html

## `torch.func`

The research roadmap uses `vmap`, `jvp`, `vjp`, `jacrev`, `jacfwd` and `functional_call` for vectorised screening and batched intervention studies. The current release uses conventional graph execution for broad compatibility; vectorised intervention compilation remains planned work.

Official reference:

* https://docs.pytorch.org/docs/2.13/func.html

## Autograd

VRAVEN runs an explicit reverse-mode gradient probe on the captured graph because export is performed under `no_grad` and static tensor metadata alone cannot establish gradient availability.

Official reference:

* https://docs.pytorch.org/docs/2.13/autograd.html

## Profiler and compiler provenance

The optional profiler records per-node execution time in the captured graph. A future research branch will connect high-level concepts to TorchInductor provenance and generated kernels.

Official references:

* https://docs.pytorch.org/docs/2.13/profiler.html
* https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_inductor_provenance.html

## Custom operators

Unknown custom operators remain unresolved unless an operator-semantics contract is registered. The contract should eventually include fake-tensor behaviour, differentiation, vectorisation and intervention semantics.

Official reference:

* https://docs.pytorch.org/docs/2.13/library.html

## Important boundary

Understanding PyTorch recovers computation. It does not automatically recover the meaning of a learned representation. VRAVEN separates:

```text
computation capture
representation discovery
causal mechanism testing
semantic grounding
human explanation
```
