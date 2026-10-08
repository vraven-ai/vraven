# VRAVEN Compiler-Decompiler Research Source Index

## Purpose

This index records the primary technical and research sources that informed
VRAVEN's compiler-assisted causal-decompilation design. It does not redistribute
publisher-controlled papers or books. It provides official documentation,
open-access papers and research pages so that a developer can reproduce the
design review locally.

## PyTorch 2.13 execution and compiler foundation

| Source | VRAVEN use |
| --- | --- |
| https://docs.pytorch.org/docs/2.13/index.html | API stability policy and top-level subsystem map |
| https://docs.pytorch.org/docs/2.13/generated/torch.nn.Module.html | Parameters, buffers, state, hierarchy and fallback instrumentation |
| https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/export.html | Preferred ahead-of-time capture of normalised tensor computation |
| https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/export/ir_spec.html | Export graph signatures, ATen graph form and constraints |
| https://docs.pytorch.org/docs/2.13/fx.html | Graph interpretation, rewriting and executable graph modules |
| https://docs.pytorch.org/docs/2.13/autograd.html | Reverse-mode derivative paths and local sensitivity screening |
| https://docs.pytorch.org/docs/2.13/func.html | `vmap`, JVP, VJP, Jacobian, Hessian and functional transforms |
| https://docs.pytorch.org/docs/2.13/generated/torch.func.functional_call.html | Functional execution under supplied parameters and buffers |
| https://docs.pytorch.org/docs/2.13/generated/torch.func.functionalize.html | Mutation and view removal before graph transformation |
| https://docs.pytorch.org/docs/2.13/profiler.html | Operator time, memory, source and shape evidence |
| https://docs.pytorch.org/docs/2.13/library.html | Custom operator, fake tensor, autograd and vectorisation contracts |
| https://docs.pytorch.org/docs/2.13/meta.html | Allocation-free structural reconnaissance and its limitations |
| https://docs.pytorch.org/docs/2.13/user_guide/torch_compiler/torch.compiler_inductor_provenance.html | Planned mapping from explanation concepts to generated kernels |

## Mechanistic interpretability and causal faithfulness

| Source | Relevance to VRAVEN |
| --- | --- |
| Geiger et al. (2025), *Causal Abstraction: A Theoretical Foundation for Mechanistic Interpretability*, JMLR. https://jmlr.org/papers/v26/23-0058.html | Formal basis for evaluating whether a high-level mechanism is faithful under interventions |
| Gupta et al. (2024), *InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques*, NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2024/hash/a8f7d43ae092d9a5295775eb17f3f4f7-Abstract-Datasets_and_Benchmarks_Track.html | Motivates known-mechanism evaluation rather than relying only on natural-model plausibility |
| Cao et al. (2025), *NeurFlow: Interpreting Neural Networks through Neuron Groups and Functional Interactions*, ICLR. https://proceedings.iclr.cc/paper_files/paper/2025/hash/b6260ae5566442da053e5ab5d691067a-Abstract-Conference.html | Establishes that neuron-group and hierarchical interaction views are adjacent work, not sufficient novelty by themselves |
| Lindsey et al. (2024), *Sparse Crosscoders for Cross-Layer Features and Model Diffing*. https://transformer-circuits.pub/2024/crosscoders/index.html | Motivates persistent features across layers and cautions against fixed neuron identity |
| Ameisen et al. (2025), *Circuit Tracing: Revealing Computational Graphs in Language Models*. https://transformer-circuits.pub/2025/attribution-graphs/methods.html | Adjacent feature-level graph reconstruction and intervention validation |

## Programme extraction and neural decompilation

| Source | Relevance to VRAVEN |
| --- | --- |
| Michaud et al. (2024), *Opening the AI Black Box: Program Synthesis via Mechanistic Interpretability*. https://arxiv.org/abs/2402.05110 | Demonstrates executable programme extraction for algorithmic RNN tasks |
| Zhang et al. (2026), *Weights to Code: Extracting Interpretable Algorithms from the Discrete Transformer*. https://arxiv.org/abs/2601.05770 | Shows programme extraction from a deliberately discrete transformer, defining an important boundary for VRAVEN's claim |
| Sutter et al. (2025), *The Non-Linear Representation Dilemma: Is Causal Abstraction Enough for Mechanistic Interpretability?*. https://arxiv.org/abs/2507.08802 | Warns that unconstrained high-level alignments can become vacuous, motivating VRAVEN's compactness, intervention and validity constraints |

## Design consequence

The research gap is not any one of concept discovery, cross-layer tracking, circuit graphs or programme synthesis. The candidate contribution is their compiler-assisted integration across ordinary PyTorch models:

```text
normalised computation capture
    + architecture-neutral Mechanism IR
    + graph-compiled interventions
    + intervention-constrained executable programme
    + contrastive and counterfactual verification
    + explicit validity, coverage and dark-matter certificate
```

The package remains a research alpha. The source index establishes provenance, not state-of-the-art status.
