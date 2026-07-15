# Contributing

Thank you for improving scDRS-FM.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev,notebooks]'
```

## Before opening a pull request

Run at least:

```bash
python -m compileall scdrs_fm run_scdrs_fm.py run_decompose_gradients.py
scdrs-fm --help
scdrs-fm-decompose --help
```

If you add tests, run:

```bash
python -m pytest
```

## Documentation changes

Update the README and relevant files under `docs/` whenever command-line arguments, outputs, installation steps, or file formats change.
