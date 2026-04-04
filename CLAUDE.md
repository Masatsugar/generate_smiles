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
├── data/
│   └── train.txt                 # Training corpus (~380K SMILES strings)
├── molenv.py                     # Placeholder module
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

## Running Code

No package installation needed. Run modules directly:

```bash
python models/rnn_model.py            # Load pretrained model, sample 64 SMILES
python models/customs/chem_vae.py     # Train/validate VAE
python models/customs/chem_gan.py     # Train GAN
python models/customs/chemutils.py    # Test reconstruction utilities
```

## Dependencies

Python 3.6+ with: `torch`, `torchvision`, `numpy`, `pandas`. No `requirements.txt` exists — install manually. GPU optional (code checks `torch.cuda.is_available()`).

## Code Style & Linting

Pre-commit hooks are configured (`.pre-commit-config.yaml`):

- **black** (21.9b0) — formatting (target: Python 3)
- **isort** (5.10.1) — import sorting
- **flake8** (4.0.1) — linting (max-line-length=120, ignores: E203, E501, C901, W503, F401)
- **mypy** (0.910) — type checking
- Standard hooks: trailing-whitespace, check-docstring-first, check-added-large-files, detect-private-key

Install hooks: `pre-commit install`

## Naming Conventions

- **Classes**: PascalCase (`SmilesRnn`, `ChemVAE`, `Encoder`)
- **Functions/variables**: snake_case (`pad_smile`, `smiles_to_hot`)
- **Constants**: UPPERCASE (`PAD`, `BEGIN`, `END`)
- Type hints used in function signatures
- Docstrings with Args/Returns sections for public APIs

## Testing

No formal test suite exists. Modules contain `__main__` blocks that serve as usage examples and smoke tests.

## CI/CD

No CI/CD pipelines configured. No Docker setup.

## Important Notes

- The pretrained model file (`models/pretrained/model_final_0.473.pt`, ~101MB) is tracked in git. Do not re-add or duplicate large binary files.
- Training data is in `data/train.txt` (one SMILES per line, ~9.6MB).
- The `.gitignore` excludes `/tmp/`, `/upload/`, virtual envs, and standard Python artifacts.
- No `requirements.txt` or `pyproject.toml` — dependency versions are not pinned.
