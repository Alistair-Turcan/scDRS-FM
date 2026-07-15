# Installation

## Requirements

- Python 3.9 or newer.
- A working C/C++/Fortran build toolchain may be required by scientific Python dependencies on some platforms.
- Enough memory for dense scoring operations. Large `.h5ad` files can require substantial RAM.

## Recommended environment

```bash
git clone https://github.com/<your-org>/scDRS-FM.git
cd scDRS-FM
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

This installs two console scripts:

```bash
scdrs-fm --help
scdrs-fm-decompose --help
```

## Conda/mamba environment

If your platform has difficulty compiling scientific dependencies, create a conda environment first and then install the repository:

```bash
mamba create -n scdrs-fm python=3.10 numpy pandas scipy scanpy statsmodels tqdm -c conda-forge -y
mamba activate scdrs-fm
python -m pip install scdrs magic-impute scprep
python -m pip install -e .
```

## Development install

```bash
python -m pip install -e '.[dev,notebooks]'
```

## Verify installation

```bash
python - <<'PY'
import scdrs_fm
print(scdrs_fm.__all__)
PY
scdrs-fm --help
scdrs-fm-decompose --help
```

## Notes about optional methods

- `--imputation magic` requires `magic-impute` and `scprep`.
- `--imputation alra` depends on functionality available through Scanpy/external single-cell tooling in your environment.
- `--imputation knn` uses the implementation in `scdrs_fm.data_processing` and standard scientific Python dependencies.
