# CLAUDE.md

## Project Overview

**generate_smiles** — Generative deep learning models for molecular SMILES string generation in PyTorch. Implements RNN/LSTM language models, Variational Autoencoders (VAE), and Generative Adversarial Networks (GAN) for drug-like molecule design.

## Repository Structure

```
generate_smiles/
├── models/
│   ├── smiles_char_dict.py       # SMILES tokenizer & character dictionary (47-char alphabet)
│   ├── rnn_model.py              # LSTM language model + Actor-Critic RL wrapper
│   ├── pretrained/               # Pretrained LSTM weights (~101MB) and config JSON
│   └── customs/
│       ├── chem_vae.py           # Conv1d encoder + GRU decoder VAE
│       ├── chem_gan.py           # Generator/Discriminator GAN
│       └── chemutils.py          # Shared utilities (one-hot encoding, padding, custom layers)
├── tests/
│   ├── test_smiles_char_dict.py  # SmilesCharDictionary tests
│   ├── test_chemutils.py         # Utility function and layer tests
│   └── test_models.py            # Model instantiation and forward pass tests
├── data/
│   └── train.txt                 # Training corpus (~380K SMILES strings)
├── pyproject.toml                # Project config, dependencies, tool settings
├── .flake8                       # Flake8 config
├── .pre-commit-config.yaml       # Pre-commit hooks config
└── README.md
```

## Key Modules

- **`models/smiles_char_dict.py`** — `SmilesCharDictionary`: tokenization, encoding/decoding multi-char tokens (e.g., "Br" → "Y"), forbidden symbol validation.
- **`models/rnn_model.py`** — `SmilesRnn` (3-layer LSTM, hidden 1024), `SmilesRnnActorCritic` (RL wrapper). Use `load_model()` to load pretrained weights.
- **`models/customs/chem_vae.py`** — `ChemVAE` with Conv1d `Encoder` and GRU `Decoder`, reparameterization trick.
- **`models/customs/chem_gan.py`** — `Generator` (Linear+GRU) and `Discriminator` (Conv1d).
- **`models/customs/chemutils.py`** — `smiles_to_hot()`, `hot_to_smiles()`, `pad_smile()`, `TimeDistributed`, `Repeat` layers, `dotdict`.

## Setup & Running

Uses **uv** for dependency management. Python 3.10+.

```bash
uv sync --group dev     # Install all dependencies including dev tools
uv run python -m pytest # Run tests
```

Run modules directly:

```bash
uv run python models/rnn_model.py            # Load pretrained model, sample 64 SMILES
uv run python models/customs/chem_vae.py     # Train/validate VAE
uv run python models/customs/chem_gan.py     # Train GAN
uv run python models/customs/chemutils.py    # Test reconstruction utilities
```

## Dependencies

Defined in `pyproject.toml`. Core: `torch>=2.0`, `torchvision>=0.15`, `numpy>=1.24`, `pandas>=2.0`. Dev: `pytest`, `black`, `isort`, `flake8`, `mypy`, `pre-commit`. GPU optional (code checks `torch.cuda.is_available()`).

## Testing

Run the test suite with pytest:

```bash
uv run python -m pytest tests/ -v
```

38 tests covering:
- `test_smiles_char_dict.py` — encoding/decoding, allowed symbols, matrix conversion
- `test_chemutils.py` — padding, one-hot encoding roundtrip, dotdict, Repeat, TimeDistributed
- `test_models.py` — forward passes for SmilesRnn, ActorCritic, Encoder, Decoder, ChemVAE, Generator, Discriminator

## Code Style & Linting

Pre-commit hooks are configured (`.pre-commit-config.yaml`):

- **black** — formatting (line-length=120, target: Python 3.10+)
- **isort** — import sorting (black profile)
- **flake8** — linting (max-line-length=120, ignores: E203, E501, C901, W503, F401)
- **mypy** — type checking
- Standard hooks: trailing-whitespace, check-docstring-first, check-added-large-files, detect-private-key

Install hooks: `pre-commit install`

## Naming Conventions

- **Classes**: PascalCase (`SmilesRnn`, `ChemVAE`, `Encoder`)
- **Functions/variables**: snake_case (`pad_smile`, `smiles_to_hot`)
- **Constants**: UPPERCASE (`PAD`, `BEGIN`, `END`)
- Type hints used in function signatures
- Docstrings with Args/Returns sections for public APIs

## Important Notes

- The pretrained model file (`models/pretrained/model_final_0.473.pt`, ~101MB) is tracked in git. Do not re-add or duplicate large binary files.
- Training data is in `data/train.txt` (one SMILES per line, ~9.6MB).
- The `.gitignore` excludes `/tmp/`, `/upload/`, virtual envs, and standard Python artifacts.
- Imports in `customs/` modules use absolute paths (e.g., `from models.customs.chemutils import ...`). Run scripts from the project root.
