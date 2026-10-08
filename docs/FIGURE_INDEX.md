# Figure index

| File key | Category | Purpose |
|---|---|---|
| `01_system_architecture` | Architecture | End-to-end route from model artefact or live model to instrumentation, on-device reduction, `NeuralTrace`, reasoning analyses and outputs. |
| `02_runtime_dataflow` | Architecture | Execution sequence and separation of transient tensors from persistent compact traces. |
| `03_package_architecture` | Architecture | Internal package boundaries, plugin points and optional Rust or GPU acceleration. |
| `04_device_execution` | Architecture | Device-following execution and bounded transfer of compact summaries to host memory. |
| `05_explanation_modes` | Methodology | Operational and semantic distinction between fast, gradient and causal modes. |
| `06_evidence_reconstruction` | Methodology | Layerwise activation, signed interaction and uncertainty channels used to construct an internal evidence graph. |
| `07_reasoning_dna_pipeline` | Methodology | Deterministic conversion of a layer-metric matrix into a fixed-width reasoning fingerprint. |
| `08_latent_community_discovery` | Methodology | Unit-profile similarity, community discovery, concept candidacy and intervention validation. |
| `09_training_monitoring` | Methodology | Training-time hooks, checkpoint traces and longitudinal neural diagnostics. |
| `10_architecture_adapters` | Methodology | Mapping architecture-specific internal signals into a common `NeuralTrace` representation. |
| `11_benchmark_protocol` | Validation | Controlled comparison against SHAP, LIME and relevant gradient or architecture-specific methods. |
| `12_causal_validation_loop` | Validation | Candidate selection, intervention, re-inference, matched controls and signed effect estimation. |
| `13_reliability_calibration` | Validation | Multidimensional reliability reporting and the requirements for any calibrated composite. |
| `14_reasoning_drift` | Validation | Comparison of fingerprint distributions across time periods, datasets or domains. |
| `15_reproducibility_provenance` | Validation | Metadata, hashes, configuration and artefacts required to reproduce a trace and figure. |
| `16_visual_contract` | Visual system | Relationship between measured data, visual encodings, standalone figures and dual render modes. |
| `17_user_workflow` | Integration | Local installation, live-model use, safe state-dictionary loading and report output. |

## Status notation

* Solid boundary: implemented architectural component or defined methodological stage.
* Dashed boundary: planned adapter, optional acceleration path or step that requires external validation.
* Coloured edge: semantic type of information or evidence.
* Grey edge: transport, ordering or neutral control flow.
