# AGENTS.md — sdypy-excitation contributor & agent guide

Guidance for anyone (human or AI) developing **sdypy-excitation**. Claude Code
reads it via a one-line `CLAUDE.md` (`@AGENTS.md`). The org-wide rules
(development workflow, SEP governance, naming, canonical specs) live in the
sdypy hub and are linked at the end, never restated here.

## What this package is

Excitation signals: the `excitation` portion of the `sdypy` namespace (PEP 420,
`from sdypy import excitation`). It is a shim: it holds no signal-generation
code, only `sdypy/excitation/__init__.py`, which re-exports eleven functions
from the backend library [pyExSi](https://github.com/ladisk/pyExSi) (random,
burst, impulse, sine-sweep, PSD-based and non-Gaussian signals, plus the
`get_psd` and `get_kurtosis` helpers) and lists them in `__all__`. Changes to the signal-generation code belong in pyExSi. Because it
is a shim, this repo carries no OpenSpec configuration of its own; the hub's
`sibling-package-template` spec exempts it.

## Development environment

Use `uv`. Supported Python versions: `requires-python` in `pyproject.toml`.

```console
uv venv && uv pip install -e ".[dev]"   # dev install
pytest                                  # tests
python -m build                         # sdist + wheel
```

## Org-wide rules (sdypy hub)

Copied verbatim from the hub's `AGENTS.md`; edit it there. The hub's template
checker compares this copy.

<!-- >>> sdypy hub links -->
- [AGENTS.md](https://github.com/sdypy/sdypy/blob/main/AGENTS.md) — workflow and definition of done
- [AGENTS.md § Releasing](https://github.com/sdypy/sdypy/blob/main/AGENTS.md#releasing) — the only way to release
- [docs/seps/](https://github.com/sdypy/sdypy/tree/main/docs/seps/) — SEP governance docs
- [docs/source/dev/nomenclature.rst](https://github.com/sdypy/sdypy/blob/main/docs/source/dev/nomenclature.rst) — naming a new public term (SEP 2)
- [openspec/specs/](https://github.com/sdypy/sdypy/tree/main/openspec/specs/) — canonical capability specs
- [REQUIREMENTS.md](https://github.com/sdypy/sdypy/blob/main/REQUIREMENTS.md) — requirements roster
<!-- <<< sdypy hub links -->
