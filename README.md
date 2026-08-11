# Blazar Hadronic Modeling

Photo-hadronic (proton-photon, p-γ) secondary particle production model for
blazar jets, implementing the analytical parametrization of Kelner & Aharonian
(2008). Computes the gamma-ray, electron+positron, and all-flavor neutrino
SEDs produced when a relativistic proton population interacts with a target
photon field. Applied here to a TXS 0506+056-like configuration, but the
model is source-agnostic.

## Overview

In hadronic blazar jet models, relativistic protons accelerated in the jet
interact with ambient photon fields (synchrotron photons, external
disk/broad-line-region radiation, etc.) via photopion production
(p + γ → p/n + π). The resulting charged and neutral pions decay into
gamma-rays, electrons/positrons, and neutrinos of all flavors — a
characteristic multi-messenger signature that is not present in purely
leptonic (synchrotron + SSC) models.

This repository implements the semi-analytical parametrization of these
secondary particle spectra from:

- Kelner, S. R., & Aharonian, F. A. (2008). *Energy spectra of gamma rays,
  electrons, and neutrinos produced at interactions of relativistic protons
  with low energy radiation.* PRD, 78, 034013.
  [arXiv:0803.0688](https://arxiv.org/abs/0803.0688)

Given a proton injection spectrum and a target photon field (in the frame
where the proton spectrum is defined), the pipeline computes the seven
secondary channels (γ, e⁻, e⁺, νμ, ν̄μ, νe, ν̄e) and combines them into
three physically observable SEDs: gamma-rays, electrons+positrons, and
neutrinos (already including the νμ⇄νe⇄ντ oscillation-averaged flavor mix).

## Repository structure

```
blazar-hadronic-modeling/
├── kelner_pgamma_model.py     # Core physics: F(eta, x) for all 7 secondary channels
├── sed_pipeline.py             # SED integration pipeline (per-proton-energy -> final SED)
├── run_example.py              # Example script: TXS 0506+056-like configuration
├── requirements.txt
├── *param.txt                  # Kelner & Aharonian (2008) tabulated fit coefficients (7 files)
├── field.txt                   # Example target photon field (energy_eV; density), optional
└── README.md
```

**`kelner_pgamma_model.py`** implements `calculate_F()`, the dimensionless
spectral function F(η, x) for each secondary species, together with
`load_fit_tables()` (loads the tabulated Kelner & Aharonian coefficients
once) and `proton_spectrum()` (the primary proton injection spectrum, power
law with an optional exponential cutoff).

**`sed_pipeline.py`** implements the three-stage integration: for each proton
energy bin, integrate F(η, x) over the target photon field
(`integrate_single_proton_energy`), remap the result onto shared output
energy grids (`_remap_to_final_grid`), then integrate over the proton
injection spectrum (`_integrate_over_proton_spectrum`). The public entry
point is `compute_photohadronic_seds()`.

**`run_example.py`** builds a Doppler-boosted external photon field and a
proton power-law spectrum, runs the pipeline, prints the jet energetics
(proton energy density, luminosity), and plots the resulting SEDs.

## Installation

Requires Python 3.10+ (for the `dataclasses` / built-in generic type hints
used throughout). The seven `*param.txt` fit-coefficient files must be kept
in the same directory passed to `load_fit_tables()` (the repository root, by
default).

## Data format

**Fit coefficient tables** (`gammaparam.txt`, `positronparam.txt`,
`electronparam.txt`, `numuparam.txt`, `antinumuparam.txt`, `nueparam.txt`,
`antinueparam.txt`): semicolon-delimited, no header, four columns:

```
eta/eta0 ; s ; delta ; B
```

These are the tabulated Kelner & Aharonian (2008) fit coefficients and
shouldn't need to change unless you are working from an updated
parametrization.

**Target photon field** (e.g. `field.txt`): semicolon-delimited, no header,
two columns:

```
energy_eV ; number_density
```

where `number_density` is the differential photon number density
[cm⁻³ eV⁻¹], in the same frame as the proton spectrum (typically the
emitting-region comoving frame — boost any external field into that frame
before passing it in, as `run_example.py` does).

## Usage

```bash
python run_example.py
```

Edit the configuration block at the top of `run_example.py` (source name,
jet parameters, target photon field, proton spectrum) for a different
source. To use the pipeline directly:

```python
from kelner_pgamma_model import load_fit_tables
from sed_pipeline import SimulationGridConfig, TargetPhotonField, compute_photohadronic_seds

fit_tables = load_fit_tables(".")               # loads the 7 *param.txt tables once
target_field = TargetPhotonField.from_file("field.txt")

result = compute_photohadronic_seds(
    proton_energy_min_eV=1.0e14,
    proton_energy_max_eV=6.0e14,
    spectral_index=2.0,
    cutoff_energy_eV=None,                       # or a cutoff energy in eV
    target_field=target_field,
    fit_tables=fit_tables,
    grid=SimulationGridConfig(),                 # defaults match the reference implementation
)

result.save(".")   # writes gamma_SED.txt, electron_SED.txt, neutrino_SED.txt
```


## References

- Kelner, S. R., & Aharonian, F. A. (2008). *Energy spectra of gamma rays,
  electrons, and neutrinos produced at interactions of relativistic protons
  with low energy radiation.* PRD, 78, 034013.
  [arXiv:0803.0688](https://arxiv.org/abs/0803.0688)
- Dermer, C. D., & Menon, G. (2009). *High Energy Radiation from Black Holes:
  Gamma Rays, Cosmic Rays, and Neutrinos.* Princeton University Press.
