"""
Example: photo-hadronic (p-gamma) secondary particle SEDs for a blazar jet.

Sets up an external (accretion-disk-like) target photon field boosted into
the emitting-region frame, a primary proton power-law spectrum, and computes
the resulting gamma-ray, electron+positron, and neutrino SEDs via
`sed_pipeline.compute_photohadronic_seds`.

Parameters below (doppler factor, redshift, blob radius) match a TXS 0506+056-
like configuration; adjust `SOURCE_NAME` and the parameter block for a
different source.

Usage
-----
    python run_example.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import simpson

from kelner_pgamma_model import load_fit_tables
from sed_pipeline import SimulationGridConfig, TargetPhotonField, compute_photohadronic_seds

EV_TO_ERG = 1.60218e-12  # erg per eV
SPEED_OF_LIGHT = 2.99792458e10  # cm/s

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SOURCE_NAME = "TXS 0506+056"

# Jet / emitting-region parameters
DOPPLER_FACTOR = 30.0
BULK_LORENTZ_FACTOR = DOPPLER_FACTOR / 2.0
BLOB_RADIUS_CM = 1.0e17
REDSHIFT = 0.3365
LUMINOSITY_DISTANCE_CM = 5.643309e27
MAGNETIC_FIELD_GAUSS = 69.0

# External target photon field (accretion-disk-like power law), defined in the
# AGN rest frame and then boosted into the blob frame by the bulk Lorentz factor
EXTERNAL_FIELD_ENERGY_MIN_EV = 1.33
EXTERNAL_FIELD_ENERGY_MAX_EV = 10.0
EXTERNAL_FIELD_SPECTRAL_INDEX = 2.0
EXTERNAL_FIELD_NORMALIZATION = 3.0e3  # eV^-1 cm^-3, before boosting

# Primary proton spectrum
PROTON_ENERGY_MIN_EV = 1.0e14
PROTON_ENERGY_MAX_EV = 6.0e14
PROTON_SPECTRAL_INDEX = 2.0
PROTON_NORMALIZATION_EV = 2.5e65  # eV^-1, i.e. C_p in dN/dE = C_p * (E/1eV)^-index
PROTON_CUTOFF_ENERGY_EV = None  # no exponential cutoff

# SED integration grid (see `SimulationGridConfig` for a description of each
# field; defaults below match the original reference implementation)
GRID = SimulationGridConfig()

PARAM_TABLE_DIRECTORY = "."  # directory containing the *param.txt fit tables


def build_boosted_external_field() -> TargetPhotonField:
    """Build an external power-law photon field and boost it into the blob frame.

    The field is defined as a power law in the external (e.g. accretion disk
    or broad-line region) rest frame, then Doppler-boosted into the emitting
    region's comoving frame using the standard external-radiation-field
    transformation for a blob moving with bulk Lorentz factor `Gamma`
    (see e.g. Dermer & Menon 2009, ch. 7).

    Returns
    -------
    TargetPhotonField
        The photon field in the blob's comoving frame.
    """
    energy_external_eV = np.logspace(
        np.log10(EXTERNAL_FIELD_ENERGY_MIN_EV), np.log10(EXTERNAL_FIELD_ENERGY_MAX_EV), GRID.n_photon_field_bins
    )
    energy_blob_eV = (4.0 / 3.0) * BULK_LORENTZ_FACTOR * energy_external_eV
    boost_factor = 2.0 * (4.0 / 3.0) * BULK_LORENTZ_FACTOR**2

    density_external = _power_law(energy_external_eV, EXTERNAL_FIELD_SPECTRAL_INDEX, EXTERNAL_FIELD_NORMALIZATION)
    density_blob = density_external * boost_factor

    return TargetPhotonField(energy_eV=energy_blob_eV, number_density=density_blob)


def _power_law(energy_eV: np.ndarray, index: float, normalization: float) -> np.ndarray:
    """Simple power-law shape normalization * (energy / 1 eV)^-index."""
    reference_energy_eV = 1.0
    return normalization * np.power(energy_eV / reference_energy_eV, -index)


def report_proton_energetics(target_field: TargetPhotonField) -> None:
    """Print the proton energy density, jet-frame luminosity, and magnetic field density.

    Purely informational (matches the diagnostic prints in the original
    reference implementation); does not affect the SED calculation.
    """
    proton_energy_eV = np.logspace(
        np.log10(PROTON_ENERGY_MIN_EV), np.log10(PROTON_ENERGY_MAX_EV), 100
    )
    proton_shape = _power_law(proton_energy_eV, PROTON_SPECTRAL_INDEX, PROTON_NORMALIZATION_EV)

    proton_energy_density_eV = simpson(proton_shape * proton_energy_eV, proton_energy_eV) / (
        (4.0 / 3.0) * np.pi * BLOB_RADIUS_CM**3
    )
    proton_energy_density_erg = proton_energy_density_eV * EV_TO_ERG

    proton_luminosity_eV = (
        np.pi * BLOB_RADIUS_CM**2 * SPEED_OF_LIGHT * BULK_LORENTZ_FACTOR**2 * proton_energy_density_eV
    )
    proton_luminosity_erg = proton_luminosity_eV * EV_TO_ERG

    magnetic_energy_density_erg = MAGNETIC_FIELD_GAUSS**2 / (8.0 * np.pi)

    print(f"Proton energy density in the blob:        {proton_energy_density_erg:.6e} erg/cm^3")
    print(f"Observable proton luminosity (lab frame):  {proton_luminosity_erg:.6e} erg/s")
    print(f"Magnetic field energy density in the blob: {magnetic_energy_density_erg:.6e} erg/cm^3")


def plot_seds(result, output_path: str = "photohadronic_sed.png") -> None:
    """Plot the gamma-ray, electron+positron, and neutrino SEDs on shared axes."""
    fig, ax = plt.subplots()

    ax.plot(
        result.neutrino_energy_eV, result.neutrino_sed, "-.", linewidth=2, color="red",
        label="Neutrino + Antineutrino",
    )
    ax.plot(
        result.electron_energy_eV, result.electron_sed, "--", linewidth=2, color="blue",
        label="Electron + Positron",
    )
    ax.plot(result.gamma_energy_eV, result.gamma_sed, "-", linewidth=2, color="black", label="Gamma-rays")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("E (eV)")
    ax.set_ylabel("E/t (eV/s)")
    ax.grid(linestyle="--", linewidth=0.5)
    ax.legend(loc="best", numpoints=1)

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)


def main() -> None:
    print(f"Photo-hadronic secondary particle SEDs for {SOURCE_NAME}")
    fit_tables = load_fit_tables(PARAM_TABLE_DIRECTORY)
    target_field = build_boosted_external_field()

    report_proton_energetics(target_field)

    result = compute_photohadronic_seds(
        proton_energy_min_eV=PROTON_ENERGY_MIN_EV,
        proton_energy_max_eV=PROTON_ENERGY_MAX_EV,
        spectral_index=PROTON_SPECTRAL_INDEX,
        cutoff_energy_eV=PROTON_CUTOFF_ENERGY_EV,
        target_field=target_field,
        fit_tables=fit_tables,
        grid=GRID,
    )

    result.save(".")
    plot_seds(result)


if __name__ == "__main__":
    main()
