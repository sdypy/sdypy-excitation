"""sdypy.excitation - excitation signal generation (random, burst, sine-sweep, PSD-based, non-Gaussian), re-exported from pyExSi under the sdypy namespace."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("sdypy-excitation")
except PackageNotFoundError:  # source checkout without installed metadata
    __version__ = "0+unknown"

from pyExSi import (burst_random, get_kurtosis, get_psd, impulse, nonstationary_signal, normal_random, pseudo_random, random_gaussian, sine_sweep, stationary_nongaussian_signal, uniform_random)

__all__ = ["burst_random", "get_kurtosis", "get_psd", "impulse", "nonstationary_signal", "normal_random", "pseudo_random", "random_gaussian", "sine_sweep", "stationary_nongaussian_signal", "uniform_random"]
