"""Tests for the OHLCV loader: error handling, retries, cleaning, fallbacks.

yfinance is mocked throughout — no network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slippage.data import DataDownloadError
from slippage.data import loader as loader_mod


@pytest.fixture(autouse=True)
def _isolate_cache(tmp_path, monkeypatch):
    """Point the parquet cache at a temp dir and make retries instant."""
    monkeypatch.setattr(loader_mod, "RAW_DIR", tmp_path)
    monkeypatch.setattr(loader_mod.time, "sleep", lambda *_: None)


def _raw(n=600):
    idx = pd.date_range("2024-01-02 09:30", periods=n, freq="1h", tz="UTC")
    base = 100 + np.arange(n) * 0.01
    return pd.DataFrame(
        {"Open": base, "High": base + 1, "Low": base - 1, "Close": base,
         "Volume": np.full(n, 1_000_000.0)},
        index=idx,
    )


def test_fetch_one_empty_returns_none(monkeypatch):
    monkeypatch.setattr(loader_mod.yf, "download", lambda *a, **k: pd.DataFrame())
    assert loader_mod._fetch_one("XXX", "2024-01-01", "2024-02-01", "1h") is None


def test_fetch_one_network_error_retries_then_raises(monkeypatch):
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise ConnectionError("network down")

    monkeypatch.setattr(loader_mod.yf, "download", boom)
    with pytest.raises(DataDownloadError):
        loader_mod._fetch_one("XXX", "2024-01-01", "2024-02-01", "1h")
    assert calls["n"] == 3  # retried


def test_fetch_one_missing_columns_raises(monkeypatch):
    bad = _raw(10).drop(columns=["Volume"])
    monkeypatch.setattr(loader_mod.yf, "download", lambda *a, **k: bad)
    with pytest.raises(DataDownloadError):
        loader_mod._fetch_one("XXX", "2024-01-01", "2024-02-01", "1h")


def test_fetch_one_success_caches(monkeypatch, tmp_path):
    monkeypatch.setattr(loader_mod.yf, "download", lambda *a, **k: _raw(600))
    df = loader_mod._fetch_one("AAA", "2024-01-01", "2024-06-01", "1h")
    assert df is not None
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert any(tmp_path.iterdir())  # parquet cached


def test_clean_drops_zero_volume_and_nan():
    df = pd.DataFrame({
        "open": [1.0, 2.0, np.nan, 4.0],
        "high": [1.0, 2.0, 3.0, 4.0],
        "low": [1.0, 2.0, 3.0, 4.0],
        "close": [1.0, 2.0, 3.0, 4.0],
        "volume": [100.0, 0.0, 100.0, 100.0],
    }, index=pd.date_range("2024-01-01", periods=4, freq="1h", tz="UTC"))
    cleaned = loader_mod._clean(df)
    assert len(cleaned) == 2  # row1 zero-volume and row2 NaN-open both dropped


def test_download_ohlcv_uses_fallback(monkeypatch):
    def fake(symbol, *a, **k):
        if symbol == "PRCT":
            return _raw(10)        # too short -> triggers fallback
        return _raw(600)
    monkeypatch.setattr(loader_mod.yf, "download", fake)
    out = loader_mod.download_ohlcv(["PRCT"])
    assert "MGNI" in out  # fell back


def test_download_ohlcv_raises_when_all_empty(monkeypatch):
    monkeypatch.setattr(loader_mod.yf, "download", lambda *a, **k: pd.DataFrame())
    with pytest.raises(DataDownloadError):
        loader_mod.download_ohlcv(["AAA", "BBB"])
