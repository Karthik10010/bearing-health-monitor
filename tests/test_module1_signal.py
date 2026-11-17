"""Module 1 tests on synthetic signals (no CWRU download needed)."""
import io

import numpy as np
import pytest
import scipy.io

from pdm.data.cwru_catalog import CATALOG, by_number
from pdm.data.loader import load_upload, read_cwru_mat, resample_to
from pdm.signal.bearing import defect_frequencies
from pdm.signal.features import extract_features, feature_groups, feature_names, params_from_config
from pdm.signal.windowing import window_signal

FS = 12000
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------- catalog
def test_catalog_counts():
    counts = {}
    for f in CATALOG:
        counts[f.fault] = counts.get(f.fault, 0) + 1
    assert counts == {"Normal": 4, "Inner race": 16, "Ball": 16, "Outer race": 12}
    assert len({f.file_number for f in CATALOG}) == len(CATALOG)


def test_catalog_no_outer_028_and_rates():
    assert not [f for f in CATALOG if f.fault == "Outer race" and f.size_in == 0.028]
    assert all(f.fs == 48000 for f in CATALOG if f.fault == "Normal")
    assert all(f.fs == 12000 for f in CATALOG if f.fault != "Normal")
    assert by_number(130).name == "OR007@6_0"
    assert by_number(3001).url.endswith("/3001.mat")


# ---------------------------------------------------------------- bearing
def test_defect_frequencies_1750rpm():
    d = defect_frequencies(1750)
    assert d.shaft_hz == pytest.approx(29.1667, rel=1e-4)
    assert d.bpfo == pytest.approx(3.5848 * 1750 / 60, rel=1e-9)
    assert 104 < d.bpfo < 105.5
    with pytest.raises(ValueError):
        defect_frequencies(0)


# ---------------------------------------------------------------- loader
def _mat_bytes(d: dict) -> io.BytesIO:
    buf = io.BytesIO()
    scipy.io.savemat(buf, d)
    buf.seek(0)
    return buf


def test_read_mat_picks_matching_key_among_several():
    a, b = np.ones((100, 1)), 2 * np.ones((100, 1))
    ms = read_cwru_mat(_mat_bytes({"X098_DE_time": a, "X099_DE_time": b, "X099RPM": np.array([[1750]])}), 99)
    assert ms.key == "X099_DE_time" and np.all(ms.signal == 2) and ms.rpm == 1750
    assert any("multiple DE keys" in n for n in ms.notes)


def test_read_mat_ambiguous_fails():
    with pytest.raises(ValueError):
        read_cwru_mat(_mat_bytes({"X001_DE_time": np.ones(10), "X002_DE_time": np.ones(10)}))


def test_resample_48k_to_12k_keeps_tone_and_removes_alias():
    t = np.arange(48000) / 48000
    x = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 9000 * t)  # 9 kHz is above the new Nyquist
    y = resample_to(x, 48000, 12000)
    assert y.size == 12000
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    f = np.fft.rfftfreq(y.size, 1 / 12000)
    peak_1k = spec[np.argmin(np.abs(f - 1000))]
    alias_3k = spec[np.argmin(np.abs(f - 3000))]  # where 9 kHz would fold to without filtering
    assert alias_3k < 0.01 * peak_1k


def test_load_upload_formats():
    x = RNG.standard_normal(5000)
    buf = io.BytesIO()
    np.save(buf, x)
    assert np.allclose(load_upload(buf.getvalue(), "a.npy"), x)
    csv_text = "signal\n" + "\n".join(repr(v) for v in x.tolist())  # plain Python floats
    assert np.allclose(load_upload(io.StringIO(csv_text), "a.csv"), x)
    assert np.allclose(load_upload(_mat_bytes({"vib": x.reshape(-1, 1)}), "a.mat"), x)
    with pytest.raises(ValueError):
        load_upload(b"xx", "a.wav")


# ---------------------------------------------------------------- windowing
def test_windowing_shapes():
    w = window_signal(np.arange(10000), 2048, 0.5)
    assert w.shape == (8, 2048)
    assert w[1, 0] == 1024
    with pytest.raises(ValueError):
        window_signal(np.zeros(100), 2048, 0.5)


# ---------------------------------------------------------------- features
@pytest.fixture(scope="module")
def params():
    return params_from_config()


def _feat(x, rpm, params):
    names = feature_names(params)
    return dict(zip(names, extract_features(window_signal(x), FS, rpm, params).mean(axis=0)))


def test_feature_names_and_groups(params):
    names = feature_names(params)
    g = feature_groups(names)
    assert len(names) == len(set(names))
    assert len(g["time"]) + len(g["frequency"]) + len(g["envelope"]) == len(names)
    assert extract_features(RNG.standard_normal((2, 2048)), FS, 1750, params).shape == (2, len(names))


def test_sine_rms_and_dominant_frequency(params):
    t = np.arange(FS) / FS
    f = _feat(2.0 * np.sin(2 * np.pi * 500 * t), 1750, params)
    assert f["t_rms"] == pytest.approx(2 / np.sqrt(2), rel=0.01)
    assert f["t_crest"] == pytest.approx(np.sqrt(2), rel=0.02)
    assert f["t_kurtosis"] == pytest.approx(1.5, rel=0.05)  # pure sine has kurtosis 1.5
    assert abs(f["f_dominant"] - 500) < FS / 2048
    assert f["f_band_1"] > 0.95  # 500 Hz falls in the first 0-750 Hz band


def _impulse_train(rate_hz, seconds=2.0, resonance=3500.0):
    n = int(FS * seconds)
    x = 0.05 * RNG.standard_normal(n)
    t_imp = np.arange(0, seconds, 1 / rate_hz)
    ring = np.exp(-np.arange(120) / 15.0) * np.sin(2 * np.pi * resonance * np.arange(120) / FS)
    for ti in t_imp:
        i = int(ti * FS)
        x[i:i + ring.size] += ring[: max(0, min(ring.size, n - i))]
    return x


def test_outer_race_impulses_light_up_bpfo(params):
    rpm = 1797
    bpfo = defect_frequencies(rpm).bpfo  # ~107 Hz
    f = _feat(_impulse_train(bpfo), rpm, params)
    assert f["e_bpfo_1"] > 5
    assert f["e_bpfo_1"] > 2 * f["e_bpfi_1"]
    assert f["t_kurtosis"] > 4  # impulsive


def test_inner_race_impulses_light_up_bpfi(params):
    rpm = 1750
    bpfi = defect_frequencies(rpm).bpfi
    f = _feat(_impulse_train(bpfi), rpm, params)
    assert f["e_bpfi_1"] > 5
    assert f["e_bpfi_1"] > 2 * f["e_bpfo_1"]


def test_noise_has_no_defect_peaks(params):
    f = _feat(RNG.standard_normal(2 * FS), 1750, params)
    assert f["e_bpfo_1"] < 5 and f["e_bpfi_1"] < 5
    assert f["t_kurtosis"] == pytest.approx(3.0, abs=0.5)
