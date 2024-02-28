sdypy-excitation
================

Frequency response function as used in structural dynamics.
-----------------------------------------------------------

The ``sdypy-excitation`` is a namespace project of the ``sdypy`` framework and is a direct link
to the ``pyExSi`` package (developed at `pyExSi <https://github.com/ladisk/pyExSi>`_).

Use the ``sdypy`` package to conveniently access the functionality of the
`pyExSi <https://github.com/ladisk/pyExSi>`_ package through its namespace (see example below). 

Other functionalities of the ``sdypy`` framework include:

- `sdypy-EMA <https://github.com/sdypy/sdypy-EMA>`_: Experimental Modal Analysis
- `sdypy-io <https://github.com/sdypy/sdypy-io>`_: Input/Output operations (LVM files, UFF files)
- `sdypy-FRF <https://github.com/sdypy/sdypy-FRF>`_: Frequency Response Function estimation


For more information check out the showcase examples and see documentation_.

Basic ``sdypy-excitation`` usage:
---------------------------------

Import the module:
~~~~~~~~~~~~~~~~~~

.. code:: python

    import numpy as np

    # import the sdypy module
    import sdypy as sd

    N = 2**16 # number of data points of time signal
    fs = 1024 # sampling frequency [Hz]
    t = np.arange(0,N)/fs # time vector

    # define frequency vector and one-sided flat-shaped PSD
    M = N//2 + 1 # number of data points of frequency vector
    freq = np.arange(0, M, 1) * fs / N # frequency vector
    freq_lower = 50 # PSD lower frequency limit  [Hz]
    freq_upper = 100 # PSD upper frequency limit [Hz]
    PSD = sd.excitation.get_psd(freq, freq_lower, freq_upper) # one-sided flat-shaped PSD

    #get gaussian stationary signal
    gausian_signal = sd.excitation.random_gaussian((N, PSD, fs))

    #get non-gaussian non-stationary signal, with kurtosis k_u=10
    #amplitude modulation, modulating signal defined by PSD
    PSD_modulating = sd.excitation.get_psd(freq, freq_lower=1, freq_upper=10)
    #define array of parameters delta_m and p
    delta_m_list = np.arange(.1,2.1,.5)
    p_list = np.arange(.1,2.1,.5)
    #get signal
    nongaussian_nonstationary_signal = sd.excitation.nonstationary_signal(N,PSD,fs,k_u=5,modulating_signal=('PSD', PSD_modulating),param1_list=p_list,param2_list=delta_m_list)


.. _documentation: https://sdypy-excitation.readthedocs.io/en/latest/
