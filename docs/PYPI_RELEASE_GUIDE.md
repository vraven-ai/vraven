# Publishing VRAVEN to PyPI

VRAVEN uses PyPI Trusted Publishing from GitHub Actions. Do not create, store or
commit a long-lived PyPI API token.

## Public-release warning

PyPI is public. Publishing exposes the Python source contained in the wheel and
source distribution even when the GitHub repository remains private. A release
must be authorised only after reviewing the generated distributions.

## One-time configuration

Create a GitHub environment named `pypi`. Where the repository plan supports
deployment protection rules, require an authorised reviewer.

After the environment and PyPI publisher are configured, create the repository
Actions variable `PYPI_PUBLISH_ENABLED` with the exact value `true`. The publish
job is fail-closed and remains skipped while this variable is absent.

In the PyPI project's Publishing settings, configure its GitHub publisher:

| Field | Value |
| --- | --- |
| PyPI project name | `vraven` |
| GitHub owner | `vraven-ai` |
| Repository | `vraven` |
| Workflow filename | `release.yml` |
| Environment | `pypi` |

For a project that does not exist yet, configure the same values as a pending
publisher from the PyPI account settings. The first successful publication
creates the project and converts that pending entry into a normal publisher.

## Local release validation

Use an isolated environment with the development and packaging tools installed:

```bash
python -m venv .release-venv
source .release-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]' build twine
```

Run the release gates:

```bash
python -m compileall -q src tests examples benchmarks scripts
PYTHONPATH=src pytest -q
rm -rf build dist src/vraven.egg-info
python -m build
python -m twine check dist/*
```

Review the wheel and source distribution. They must not contain credentials,
private paths, datasets, model checkpoints, notebooks or generated reports.

## Publishing

Confirm that `pyproject.toml` and `src/vraven/__init__.py` contain the same
version. Commit and push the release-ready source before tagging it.

Create an annotated tag that exactly matches the package version with a `v`
prefix:

```bash
git tag -a v0.10.1 -m "VRAVEN 0.10.1"
git push origin v0.10.1
```

Create a draft GitHub Release from that tag and complete the final disclosure
review. Publishing the GitHub Release triggers `.github/workflows/release.yml`.
The workflow verifies the tag, builds fresh distributions, validates them and
waits at the `pypi` environment gate before using short-lived OIDC credentials
to upload.

Do not upload the locally generated files manually. GitHub Actions must build
the authoritative PyPI artifacts from the tagged commit.

## Post-publication verification

Install the exact release into a clean environment:

```bash
python -m venv /tmp/vraven-pypi-test
source /tmp/vraven-pypi-test/bin/activate
python -m pip install --upgrade pip
python -m pip install vraven==0.10.1
vraven --version
vraven doctor
vraven features
```

Users may install or upgrade to the latest release with:

```bash
python -m pip install --upgrade vraven
```

PyPI release files are immutable. If any uploaded file must change, increment
the version and publish a new release.

## Release progression

```text
0.10.x     Backwards-compatible fixes within the current pre-1.0 line
0.11.0     New pre-1.0 capabilities that may refine experimental interfaces
1.0.0      Stable API only after scientific and compatibility review
```
