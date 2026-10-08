# Contributing to VRAVEN

Thank you for helping make neural-network explanations more testable,
reproducible and honest about their limits. Contributions do not need to begin
with a new algorithm: documentation, compatibility examples, numerical tests
and minimal bug reproductions are valuable research infrastructure.

## Good ways to start

* Reproduce an existing example on another supported PyTorch architecture.
* Add a compact model bundle that exposes a compatibility gap.
* Improve a claim boundary, tutorial or API example.
* Add numerical edge-case tests for an existing metric.
* Propose a research method with a precise question and validation plan.

For substantial API or methodological work, open an issue before implementation
so the evidence contract and scope can be agreed early.

## Before editing

Read:

1. `docs/IMPLEMENTATION_STATUS.md`
2. `docs/RESEARCH_METHOD.md`
3. `docs/VISUAL_SYSTEM.md` for visual work
4. `docs/API_SPEC.md` for public API work

## Local setup

```bash
git clone https://github.com/vraven-ai/vraven.git
cd vraven
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
ruff check .
pytest -q
```

The full suite currently contains more than 200 tests. During development, run
the smallest relevant test module first, then run the complete suite before a
pull request.

## Change requirements

A user-facing change requires working code, tests, documentation and an
implementation-status update. Scientific metrics also require a mathematical
definition, assumptions, interpretation, limitations and numerical tests.

Do not commit generated reports, local traces, Python caches, build directories or environment-specific files.

## Pull request evidence

Record:

* The scientific or engineering question addressed.
* The trace fields used or added.
* The public API change.
* Tests run.
* Runtime or memory impact when relevant.
* Known limitations.
* Whether both visual modes were tested.

## Release validation

```bash
bash scripts/quality_check.sh
bash scripts/build_package.sh
```
