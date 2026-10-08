# VRAVEN Benchmarks

The benchmark compares a fair `torch.inference_mode()` baseline with VRAVEN fast and gradient modes. It performs untimed warm-up iterations, synchronises asynchronous devices at timing boundaries and can write a machine-readable JSON record.

```bash
PYTHONPATH=src python benchmarks/benchmark_capture.py \
  --warmup 3 \
  --repeats 20 \
  --batch 32 \
  --width 512 \
  --json-output benchmark.json
```

Every performance-sensitive release should record:

* Native inference latency
* VRAVEN fast mode latency
* VRAVEN gradient mode latency
* Median, mean, minimum, maximum and p90 latency
* Peak memory, when measured by a platform-specific profiler
* Trace size and number of captured modules
* CPU or accelerator device
* Model architecture, parameter count, batch size and input shape
* Framework, Python and VRAVEN versions

Initial benchmark models should include a small MLP, ResNet18, a compact CNN and a small Transformer encoder. A benchmark result applies only to its declared hardware, model and configuration. It must not be presented as a universal performance claim.
