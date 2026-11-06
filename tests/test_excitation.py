# -*- coding: utf-8 -*-
"""
Functional tests for sdypy.excitation (wraps pyExSi).

All eleven curated functions are covered:
  burst_random, get_kurtosis, get_psd, impulse, nonstationary_signal,
  normal_random, pseudo_random, random_gaussian, sine_sweep,
  stationary_nongaussian_signal, uniform_random.

Import path: ``from sdypy import excitation``
"""

import numpy as np
import pytest

from sdypy import excitation


# ---------------------------------------------------------------------------
# Shared helpers / fixtures
# ---------------------------------------------------------------------------

SEED = 42
N = 4096  # large enough for statistical tests
FS = 1000.0  # Hz

# Build a standard frequency vector and flat PSD used by several tests
M = N // 2 + 1
FREQ = np.arange(0, M) * FS / N  # [0, FS/2] inclusive
FREQ_LO = 100.0   # Hz
FREQ_HI = 200.0   # Hz
VARIANCE = 2.0

PSD_FLAT = excitation.get_psd(FREQ, FREQ_LO, FREQ_HI, variance=VARIANCE)


def _rg(seed=SEED):
    return np.random.default_rng(seed)


# ===========================================================================
# uniform_random
# ===========================================================================

class TestUniformRandom:
    """uniform_random(N, rg=None) -> array[N]"""

    def test_shape(self):
        x = excitation.uniform_random(N, rg=_rg())
        assert x.shape == (N,)

    def test_dtype_float(self):
        x = excitation.uniform_random(N, rg=_rg())
        assert np.issubdtype(x.dtype, np.floating)

    def test_all_finite(self):
        x = excitation.uniform_random(N, rg=_rg())
        assert np.all(np.isfinite(x))

    def test_bounds(self):
        """Output is normalized by max(abs), so values must be in [-1, 1]
        and the maximum absolute value must equal 1."""
        x = excitation.uniform_random(N, rg=_rg())
        assert np.all(x >= -1.0)
        assert np.all(x <= 1.0)
        assert pytest.approx(np.max(np.abs(x)), abs=1e-12) == 1.0

    def test_reproducible(self):
        x1 = excitation.uniform_random(N, rg=_rg())
        x2 = excitation.uniform_random(N, rg=_rg())
        np.testing.assert_array_equal(x1, x2)


# ===========================================================================
# normal_random
# ===========================================================================

class TestNormalRandom:
    """normal_random(N, rg=None) -> array[N]"""

    def test_shape(self):
        x = excitation.normal_random(N, rg=_rg())
        assert x.shape == (N,)

    def test_dtype_float(self):
        x = excitation.normal_random(N, rg=_rg())
        assert np.issubdtype(x.dtype, np.floating)

    def test_all_finite(self):
        x = excitation.normal_random(N, rg=_rg())
        assert np.all(np.isfinite(x))

    def test_normalized_max(self):
        """Normalized by max(abs) — peak absolute value must equal exactly 1."""
        x = excitation.normal_random(N, rg=_rg())
        assert pytest.approx(np.max(np.abs(x)), abs=1e-12) == 1.0

    def test_mean_near_zero(self):
        """For large N the sample mean of a standard-normal draw divided by
        max(abs) is still very close to 0.  Tolerance is generous (0.05)."""
        x = excitation.normal_random(N, rg=_rg())
        assert abs(np.mean(x)) < 0.05

    def test_reproducible(self):
        x1 = excitation.normal_random(N, rg=_rg())
        x2 = excitation.normal_random(N, rg=_rg())
        np.testing.assert_array_equal(x1, x2)


# ===========================================================================
# pseudo_random
# ===========================================================================

class TestPseudoRandom:
    """pseudo_random(N, rg=None) -> array[N]

    Unit amplitude spectrum + random phase => each frequency bin has the same
    magnitude in the amplitude spectrum.  After IRFFT and normalization the
    output has shape (N,) with peak absolute value == 1.
    """

    def test_shape(self):
        x = excitation.pseudo_random(N, rg=_rg())
        assert x.shape == (N,)

    def test_all_finite(self):
        x = excitation.pseudo_random(N, rg=_rg())
        assert np.all(np.isfinite(x))

    def test_normalized_max(self):
        x = excitation.pseudo_random(N, rg=_rg())
        assert pytest.approx(np.max(np.abs(x)), abs=1e-12) == 1.0

    def test_flat_amplitude_spectrum(self):
        """The RFFT magnitudes of a *single* pseudo-random signal should all
        be equal (flat spectrum) after undoing the normalization."""
        x = excitation.pseudo_random(N, rg=_rg())
        # Undo normalization: multiply back by max(abs) before IRFFT;
        # here we just check that the raw output's amplitude spectrum is flat.
        # Because the signal is normalized *after* IRFFT the magnitudes of the
        # RFFT are proportional to the original unit amplitudes, so all bins
        # must have the same magnitude.
        spectrum = np.abs(np.fft.rfft(x))
        # Exclude DC and Nyquist bins which may differ due to real-signal FFT
        mags = spectrum[1:-1]
        rel_std = np.std(mags) / np.mean(mags)
        # For a perfect pseudo-random signal rel_std should be ~0; allow small
        # numerical noise
        assert rel_std < 0.01, f"Amplitude spectrum not flat: rel_std={rel_std:.4f}"

    def test_reproducible(self):
        x1 = excitation.pseudo_random(N, rg=_rg())
        x2 = excitation.pseudo_random(N, rg=_rg())
        np.testing.assert_array_equal(x1, x2)


# ===========================================================================
# burst_random
# ===========================================================================

class TestBurstRandom:
    """burst_random(N, A, ratio, distribution, n_bursts, periodic_bursts, rg)"""

    def test_shape_single_burst(self):
        x = excitation.burst_random(N, rg=_rg())
        assert x.shape == (N,)

    def test_shape_multiple_bursts(self):
        n_bursts = 3
        x = excitation.burst_random(N, n_bursts=n_bursts, rg=_rg())
        assert x.shape == (N * n_bursts,)

    def test_all_finite(self):
        x = excitation.burst_random(N, rg=_rg())
        assert np.all(np.isfinite(x))

    def test_trailing_zeros_ratio(self):
        """With ratio=0.3, the last 70 % of each N-block must be zero."""
        ratio = 0.3
        x = excitation.burst_random(N, ratio=ratio, rg=_rg())
        n_zero = int(np.floor(N * (1 - ratio)))
        assert np.all(x[-n_zero:] == 0.0)

    def test_nonzero_burst_region(self):
        """The burst region (first ratio*N samples) must have at least one
        non-zero sample."""
        ratio = 0.5
        x = excitation.burst_random(N, ratio=ratio, rg=_rg())
        n_burst = int(np.floor(N * ratio))
        assert np.any(x[:n_burst] != 0.0)

    def test_amplitude_uniform(self):
        """For uniform distribution, amplitude A scales the peak (before
        normalization the uniform draw is already normalized, then * A)."""
        A = 5.0
        x = excitation.burst_random(N, A=A, ratio=1.0, distribution='uniform', rg=_rg())
        assert np.max(np.abs(x)) == pytest.approx(A, rel=1e-6)

    def test_dtype_float(self):
        x = excitation.burst_random(N, rg=_rg())
        assert np.issubdtype(x.dtype, np.floating)

    def test_reproducible(self):
        x1 = excitation.burst_random(N, n_bursts=2, rg=_rg())
        x2 = excitation.burst_random(N, n_bursts=2, rg=_rg())
        np.testing.assert_array_equal(x1, x2)


# ===========================================================================
# sine_sweep
# ===========================================================================

class TestSineSweep:
    """sine_sweep(time, phi, freq_start, sweep_rate/freq_stop, mode, ...)"""

    # Build a time vector: 10 s at 1000 Sa/s
    T = 10.0
    fs_sweep = 1000.0
    N_sweep = int(T * fs_sweep)
    t = np.arange(N_sweep) / fs_sweep
    F_START = 5.0    # Hz
    F_STOP = 50.0    # Hz

    def test_shape(self):
        x = excitation.sine_sweep(self.t, freq_start=self.F_START,
                                   freq_stop=self.F_STOP)
        assert x.shape == (self.N_sweep,)

    def test_all_finite(self):
        x = excitation.sine_sweep(self.t, freq_start=self.F_START,
                                   freq_stop=self.F_STOP)
        assert np.all(np.isfinite(x))

    def test_amplitude_unity(self):
        """A pure sine sweep should have values in [-1, 1]."""
        x = excitation.sine_sweep(self.t, freq_start=self.F_START,
                                   freq_stop=self.F_STOP)
        assert np.max(np.abs(x)) <= 1.0 + 1e-9

    def test_instantaneous_frequency_start(self):
        """Use freq_return=True to get the instantaneous frequency array.
        The first sample should match freq_start."""
        _, _, freq = excitation.sine_sweep(
            self.t, freq_start=self.F_START, freq_stop=self.F_STOP,
            freq_return=True
        )
        assert freq[0] == pytest.approx(self.F_START, abs=0.5)

    def test_instantaneous_frequency_end(self):
        """The last sample of the freq array should match freq_stop."""
        _, _, freq = excitation.sine_sweep(
            self.t, freq_start=self.F_START, freq_stop=self.F_STOP,
            freq_return=True
        )
        assert freq[-1] == pytest.approx(self.F_STOP, abs=0.5)

    def test_freq_return_shape(self):
        s, phi_end, freq = excitation.sine_sweep(
            self.t, freq_start=self.F_START, freq_stop=self.F_STOP,
            freq_return=True
        )
        assert s.shape == (self.N_sweep,)
        assert freq.shape == (self.N_sweep,)

    def test_freq_monotonically_increasing_linear(self):
        """For a linear up-sweep the instantaneous frequency should be
        strictly increasing."""
        _, _, freq = excitation.sine_sweep(
            self.t, freq_start=self.F_START, freq_stop=self.F_STOP,
            freq_return=True
        )
        assert np.all(np.diff(freq) >= 0)

    def test_logarithmic_mode(self):
        """Logarithmic sweep should produce a valid array of same shape."""
        x = excitation.sine_sweep(
            self.t, freq_start=self.F_START, freq_stop=self.F_STOP,
            mode='logarithmic'
        )
        assert x.shape == (self.N_sweep,)
        assert np.all(np.isfinite(x))


# ===========================================================================
# impulse
# ===========================================================================

class TestImpulse:
    """impulse(N, n_start, width, amplitude, window)"""

    def test_shape(self):
        x = excitation.impulse(N)
        assert x.shape == (N,)

    def test_all_finite(self):
        x = excitation.impulse(N)
        assert np.all(np.isfinite(x))

    def test_peak_amplitude(self):
        """The maximum value of an impulse should equal the specified amplitude."""
        amp = 3.5
        x = excitation.impulse(N, amplitude=amp)
        assert pytest.approx(np.max(x), rel=1e-6) == amp

    def test_zeros_before_n_start(self):
        """All samples before n_start must be zero."""
        n_start = 100
        x = excitation.impulse(N, n_start=n_start, width=50)
        assert np.all(x[:n_start] == 0.0)

    def test_zeros_after_pulse_end(self):
        """All samples after n_start+width must be zero."""
        n_start = 100
        width = 50
        x = excitation.impulse(N, n_start=n_start, width=width)
        assert np.all(x[n_start + width:] == 0.0)

    def test_peak_location(self):
        """The peak should fall somewhere between n_start and n_start+width."""
        n_start = 200
        width = 100
        x = excitation.impulse(N, n_start=n_start, width=width)
        peak_idx = np.argmax(x)
        assert n_start <= peak_idx < n_start + width

    def test_triangular_window(self):
        x = excitation.impulse(N, n_start=0, width=100, window='triang')
        assert x.shape == (N,)
        assert np.all(np.isfinite(x))


# ===========================================================================
# get_psd
# ===========================================================================

class TestGetPsd:
    """get_psd(freq, freq_lower, freq_upper, variance=1)"""

    def test_shape(self):
        assert PSD_FLAT.shape == (M,)

    def test_all_finite(self):
        assert np.all(np.isfinite(PSD_FLAT))

    def test_zeros_outside_band(self):
        """PSD must be zero outside [freq_lower, freq_upper]."""
        outside = (FREQ < FREQ_LO) | (FREQ > FREQ_HI)
        assert np.all(PSD_FLAT[outside] == 0.0)

    def test_flat_inside_band(self):
        """PSD must be constant (flat) inside the band."""
        inside = (FREQ >= FREQ_LO) & (FREQ <= FREQ_HI)
        values = PSD_FLAT[inside]
        assert np.std(values) == pytest.approx(0.0, abs=1e-14)

    def test_variance_integral(self):
        """The integral (sum * df) of a one-sided PSD must equal the variance.

        get_psd sets PSD[indx] = variance / bandwidth, where
        bandwidth = freq[indx][-1] - freq[indx][0].  The actual integral via
        the trapezoidal rule may differ slightly from the nominal variance
        because the frequency resolution is finite.  We check that the flat
        PSD value * the exact bandwidth equals the requested variance.
        """
        inside = (FREQ >= FREQ_LO) & (FREQ <= FREQ_HI)
        psd_val = PSD_FLAT[inside][0]
        bandwidth = FREQ[inside][-1] - FREQ[inside][0]
        assert pytest.approx(psd_val * bandwidth, rel=1e-9) == VARIANCE

    def test_amplitude_known_flat(self):
        """For variance=1 and a 10 Hz band (100-110 Hz) the PSD value should
        equal 1/10 = 0.1 (units: unit^2/Hz)."""
        freq = np.arange(0, 501, 1.0)  # 0..500 Hz, df=1 Hz
        psd = excitation.get_psd(freq, 100.0, 110.0, variance=1.0)
        inside = (freq >= 100.0) & (freq <= 110.0)
        expected = 1.0 / (110.0 - 100.0)
        np.testing.assert_allclose(psd[inside], expected, rtol=1e-9)


# ===========================================================================
# random_gaussian
# ===========================================================================

class TestRandomGaussian:
    """random_gaussian(N, PSD, fs, rg=None) -> array[N]"""

    def test_shape(self):
        x = excitation.random_gaussian(N, PSD_FLAT, FS, rg=_rg())
        assert x.shape == (N,)

    def test_all_finite(self):
        x = excitation.random_gaussian(N, PSD_FLAT, FS, rg=_rg())
        assert np.all(np.isfinite(x))

    def test_dtype_float(self):
        x = excitation.random_gaussian(N, PSD_FLAT, FS, rg=_rg())
        assert np.issubdtype(x.dtype, np.floating)

    def test_mean_near_zero(self):
        """IFFT of random phase with zero DC → mean ≈ 0."""
        x = excitation.random_gaussian(N, PSD_FLAT, FS, rg=_rg())
        assert abs(np.mean(x)) < 0.5  # generous: the PSD has zero DC energy

    def test_reproducible(self):
        x1 = excitation.random_gaussian(N, PSD_FLAT, FS, rg=_rg())
        x2 = excitation.random_gaussian(N, PSD_FLAT, FS, rg=_rg())
        np.testing.assert_array_equal(x1, x2)

    def test_variance_order_of_magnitude(self):
        """The sample variance should be within a factor of 4 of the nominal
        variance (large-N, fixed seed gives good convergence)."""
        x = excitation.random_gaussian(N, PSD_FLAT, FS, rg=_rg())
        sample_var = np.var(x)
        assert VARIANCE / 4 < sample_var < VARIANCE * 4


# ===========================================================================
# get_kurtosis
# ===========================================================================

class TestGetKurtosis:
    """get_kurtosis(signal) -> float"""

    def test_gaussian_kurtosis_near_3(self):
        """For a large Gaussian sample the kurtosis should be close to 3."""
        rg = _rg()
        gauss = rg.standard_normal(100_000)
        k = excitation.get_kurtosis(gauss)
        assert abs(k - 3.0) < 0.1, f"kurtosis={k:.4f}, expected ~3"

    def test_nongaussian_kurtosis_greater_3(self):
        """A signal generated with stationary_nongaussian_signal(k_u=5) should
        have kurtosis clearly above 3."""
        x_ng = excitation.stationary_nongaussian_signal(
            N, PSD_FLAT, FS, k_u=5, rg=_rg()
        )
        k = excitation.get_kurtosis(x_ng)
        assert k > 3.5, f"kurtosis={k:.4f}, expected > 3.5 for k_u=5 target"

    def test_scalar_output(self):
        rg = _rg()
        x = rg.standard_normal(1000)
        k = excitation.get_kurtosis(x)
        assert np.ndim(k) == 0  # scalar

    def test_deterministic(self):
        """Same input → same kurtosis."""
        rg = _rg()
        x = rg.standard_normal(5000)
        assert excitation.get_kurtosis(x) == excitation.get_kurtosis(x)


# ===========================================================================
# stationary_nongaussian_signal
# ===========================================================================

class TestStationaryNongaussian:
    """stationary_nongaussian_signal(N, PSD, fs, s_k, k_u, mean, rg)"""

    def test_shape(self):
        x = excitation.stationary_nongaussian_signal(N, PSD_FLAT, FS, rg=_rg())
        assert x.shape == (N,)

    def test_all_finite(self):
        x = excitation.stationary_nongaussian_signal(N, PSD_FLAT, FS, rg=_rg())
        assert np.all(np.isfinite(x))

    def test_dtype_float(self):
        x = excitation.stationary_nongaussian_signal(N, PSD_FLAT, FS, rg=_rg())
        assert np.issubdtype(x.dtype, np.floating)

    def test_gaussian_default_kurtosis(self):
        """With default k_u=3 (Gaussian) the kurtosis should still be close to 3."""
        x = excitation.stationary_nongaussian_signal(N, PSD_FLAT, FS, k_u=3, rg=_rg())
        k = excitation.get_kurtosis(x)
        assert abs(k - 3.0) < 1.5

    def test_higher_kurtosis_target(self):
        """Requesting k_u=5 should yield kurtosis clearly above Gaussian (>3.5)."""
        x = excitation.stationary_nongaussian_signal(N, PSD_FLAT, FS, k_u=5, rg=_rg())
        k = excitation.get_kurtosis(x)
        assert k > 3.5, f"kurtosis={k:.4f}, expected > 3.5"

    def test_mean_shift(self):
        """Setting mean=2 should shift the sample mean by approximately 2."""
        x = excitation.stationary_nongaussian_signal(
            N, PSD_FLAT, FS, k_u=3, mean=2.0, rg=_rg()
        )
        assert abs(np.mean(x) - 2.0) < 1.0

    def test_reproducible(self):
        x1 = excitation.stationary_nongaussian_signal(N, PSD_FLAT, FS, rg=_rg())
        x2 = excitation.stationary_nongaussian_signal(N, PSD_FLAT, FS, rg=_rg())
        np.testing.assert_array_equal(x1, x2)


# ===========================================================================
# nonstationary_signal
# ===========================================================================

class TestNonstationarySignal:
    """nonstationary_signal(N, PSD, fs, k_u, modulating_signal, ..., seed, SQ)

    Uses `seed` (int) for reproducibility, not `rg`.
    """
    # Build a PSD for the modulating signal in a lower frequency band
    FREQ_LO_MOD = 1.0
    FREQ_HI_MOD = 10.0
    PSD_MOD = excitation.get_psd(FREQ, FREQ_LO_MOD, FREQ_HI_MOD, variance=1.0)

    def test_shape(self):
        x = excitation.nonstationary_signal(
            N, PSD_FLAT, FS,
            modulating_signal=('PSD', self.PSD_MOD),
            seed=SEED
        )
        assert x.shape == (N,)

    def test_all_finite(self):
        x = excitation.nonstationary_signal(
            N, PSD_FLAT, FS,
            modulating_signal=('PSD', self.PSD_MOD),
            seed=SEED
        )
        assert np.all(np.isfinite(x))

    def test_dtype_float(self):
        x = excitation.nonstationary_signal(
            N, PSD_FLAT, FS,
            modulating_signal=('PSD', self.PSD_MOD),
            seed=SEED
        )
        assert np.issubdtype(x.dtype, np.floating)

    def test_unit_variance(self):
        """The output is normalized to unit variance inside the function."""
        x = excitation.nonstationary_signal(
            N, PSD_FLAT, FS,
            modulating_signal=('PSD', self.PSD_MOD),
            seed=SEED
        )
        assert pytest.approx(np.std(x), rel=0.1) == 1.0

    def test_reproducible_with_seed(self):
        """Same integer seed should produce identical outputs."""
        x1 = excitation.nonstationary_signal(
            N, PSD_FLAT, FS,
            modulating_signal=('PSD', self.PSD_MOD),
            seed=SEED
        )
        x2 = excitation.nonstationary_signal(
            N, PSD_FLAT, FS,
            modulating_signal=('PSD', self.PSD_MOD),
            seed=SEED
        )
        np.testing.assert_array_equal(x1, x2)

    def test_higher_kurtosis(self):
        """Requesting k_u=5 should produce kurtosis > 3 (non-Gaussian)."""
        x = excitation.nonstationary_signal(
            N, PSD_FLAT, FS, k_u=5,
            modulating_signal=('PSD', self.PSD_MOD),
            seed=SEED
        )
        k = excitation.get_kurtosis(x)
        assert k > 3.0, f"kurtosis={k:.4f}, expected > 3 for k_u=5 target"
