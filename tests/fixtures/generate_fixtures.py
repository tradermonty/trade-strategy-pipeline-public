"""Generate deterministic test fixtures for all phases.

Uses fixed numpy seed for reproducibility.
Generates 5 synthetic symbols with specific price patterns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
FIXTURES_DIR = Path(__file__).parent
TRADING_DAYS = 250


def _trading_dates(n: int = TRADING_DAYS) -> pd.DatetimeIndex:
    """Generate n business day dates."""
    return pd.bdate_range(start="2023-01-02", periods=n, freq="B")


def _make_syn_vcp(rng: np.random.Generator) -> pd.DataFrame:
    """SYN_VCP: Classic VCP pattern — uptrend → 3 contracting ranges → breakout."""
    dates = _trading_dates()
    n = len(dates)

    # Phase 1: uptrend (0-99)
    prices = np.zeros(n)
    prices[0] = 50.0
    for i in range(1, 100):
        prices[i] = prices[i - 1] * (1 + rng.normal(0.002, 0.008))

    # Phase 2: contraction 1 — range ~10% (100-139)
    base = prices[99]
    for i in range(100, 140):
        swing = 0.10 * base * np.sin(2 * np.pi * (i - 100) / 40)
        prices[i] = base + swing + rng.normal(0, 0.002 * base)

    # Phase 3: contraction 2 — range ~7% (140-179)
    base = prices[139]
    for i in range(140, 180):
        swing = 0.07 * base * np.sin(2 * np.pi * (i - 140) / 40)
        prices[i] = base + swing + rng.normal(0, 0.001 * base)

    # Phase 4: contraction 3 — range ~5% (180-219)
    base = prices[179]
    for i in range(180, 220):
        swing = 0.05 * base * np.sin(2 * np.pi * (i - 180) / 40)
        prices[i] = base + swing + rng.normal(0, 0.001 * base)

    # Phase 5: breakout (220-249)
    base = prices[219]
    for i in range(220, n):
        prices[i] = base * (1 + 0.005 * (i - 219)) + rng.normal(0, 0.002 * base)

    # Volume: declining during contractions, 2x surge on breakout
    volume = np.zeros(n)
    volume[:100] = rng.normal(1_000_000, 200_000, 100)
    volume[100:140] = rng.normal(800_000, 150_000, 40)
    volume[140:180] = rng.normal(600_000, 100_000, 40)
    volume[180:220] = rng.normal(400_000, 80_000, 40)
    volume[220:] = rng.normal(2_000_000, 300_000, 30)
    volume = np.clip(volume, 100_000, None)

    close = prices
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SYN_VCP",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.astype(int),
        }
    )


def _make_syn_trend(rng: np.random.Generator) -> pd.DataFrame:
    """SYN_TREND: Steady uptrend, price > SMA50 > SMA200, no VCP."""
    dates = _trading_dates()
    n = len(dates)
    prices = 100.0 * np.cumprod(1 + rng.normal(0.001, 0.005, n))
    close = prices
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.normal(500_000, 100_000, n).astype(int)
    volume = np.clip(volume, 100_000, None)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SYN_TREND",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_syn_notrend(rng: np.random.Generator) -> pd.DataFrame:
    """SYN_NOTREND: Downtrend, price < SMA200."""
    dates = _trading_dates()
    n = len(dates)
    prices = 100.0 * np.cumprod(1 + rng.normal(-0.001, 0.008, n))
    close = prices
    high = close * (1 + rng.uniform(0, 0.012, n))
    low = close * (1 - rng.uniform(0, 0.012, n))
    open_ = close * (1 + rng.normal(0, 0.004, n))
    volume = rng.normal(300_000, 80_000, n).astype(int)
    volume = np.clip(volume, 50_000, None)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SYN_NOTREND",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_syn_gapup(rng: np.random.Generator) -> pd.DataFrame:
    """SYN_GAPUP: Has a day where Open > take-profit target."""
    dates = _trading_dates()
    n = len(dates)
    prices = 80.0 * np.cumprod(1 + rng.normal(0.0005, 0.006, n))

    # Insert a gap-up at day 150
    prices[150] = prices[149] * 1.12  # 12% gap up
    for i in range(151, n):
        prices[i] = prices[150] * (1 + rng.normal(0.0003, 0.005) * (i - 150))

    close = prices
    open_ = close.copy()
    open_[150] = prices[149] * 1.12  # Open gaps above TP target
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.normal(600_000, 120_000, n).astype(int)
    volume = np.clip(volume, 100_000, None)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SYN_GAPUP",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_syn_gapdown(rng: np.random.Generator) -> pd.DataFrame:
    """SYN_GAPDOWN: Has a day where Open < stop-loss price."""
    dates = _trading_dates()
    n = len(dates)
    prices = 90.0 * np.cumprod(1 + rng.normal(0.0003, 0.007, n))

    # Insert a gap-down at day 180
    prices[180] = prices[179] * 0.88  # 12% gap down
    for i in range(181, n):
        prices[i] = prices[180] * (1 + rng.normal(-0.0002, 0.006) * (i - 180))

    close = prices
    open_ = close.copy()
    open_[180] = prices[179] * 0.88
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.normal(500_000, 100_000, n).astype(int)
    volume = np.clip(volume, 100_000, None)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SYN_GAPDOWN",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_syn_vcp_long(rng: np.random.Generator) -> pd.DataFrame:
    """SYN_VCP_LONG: 600-bar series with 2 VCP patterns for rolling detection tests."""
    n = 600
    dates = pd.bdate_range(start="2021-01-04", periods=n, freq="B")
    prices = np.zeros(n)

    # === VCP #1: bars 0-249 ===
    # Uptrend (0-79)
    prices[0] = 50.0
    for i in range(1, 80):
        prices[i] = prices[i - 1] * (1 + rng.normal(0.002, 0.008))

    # Contraction 1: range ~10% (80-119)
    base = prices[79]
    for i in range(80, 120):
        swing = 0.10 * base * np.sin(2 * np.pi * (i - 80) / 40)
        prices[i] = base + swing + rng.normal(0, 0.002 * base)

    # Contraction 2: range ~7% (120-159)
    base = prices[119]
    for i in range(120, 160):
        swing = 0.07 * base * np.sin(2 * np.pi * (i - 120) / 40)
        prices[i] = base + swing + rng.normal(0, 0.001 * base)

    # Contraction 3: range ~5% (160-199)
    base = prices[159]
    for i in range(160, 200):
        swing = 0.05 * base * np.sin(2 * np.pi * (i - 160) / 40)
        prices[i] = base + swing + rng.normal(0, 0.001 * base)

    # Breakout #1 (200-249)
    base = prices[199]
    for i in range(200, 250):
        prices[i] = base * (1 + 0.005 * (i - 199)) + rng.normal(0, 0.002 * base)

    # === Flat zone: bars 250-299 ===
    base = prices[249]
    for i in range(250, 300):
        prices[i] = base * (1 + rng.normal(0, 0.003))

    # === VCP #2: bars 300-549 ===
    # Uptrend (300-379)
    base = prices[299]
    for i in range(300, 380):
        prices[i] = prices[i - 1] * (1 + rng.normal(0.002, 0.008))

    # Contraction 1: range ~10% (380-419)
    base = prices[379]
    for i in range(380, 420):
        swing = 0.10 * base * np.sin(2 * np.pi * (i - 380) / 40)
        prices[i] = base + swing + rng.normal(0, 0.002 * base)

    # Contraction 2: range ~7% (420-459)
    base = prices[419]
    for i in range(420, 460):
        swing = 0.07 * base * np.sin(2 * np.pi * (i - 420) / 40)
        prices[i] = base + swing + rng.normal(0, 0.001 * base)

    # Contraction 3: range ~5% (460-499)
    base = prices[459]
    for i in range(460, 500):
        swing = 0.05 * base * np.sin(2 * np.pi * (i - 460) / 40)
        prices[i] = base + swing + rng.normal(0, 0.001 * base)

    # Breakout #2 (500-549)
    base = prices[499]
    for i in range(500, 550):
        prices[i] = base * (1 + 0.005 * (i - 499)) + rng.normal(0, 0.002 * base)

    # Tail (550-599)
    base = prices[549]
    for i in range(550, n):
        prices[i] = base * (1 + rng.normal(0, 0.003))

    # Volume: declining during contractions, surge on breakouts
    volume = np.zeros(n)
    volume[:80] = rng.normal(1_000_000, 200_000, 80)
    volume[80:120] = rng.normal(800_000, 150_000, 40)
    volume[120:160] = rng.normal(600_000, 100_000, 40)
    volume[160:200] = rng.normal(400_000, 80_000, 40)
    volume[200:250] = rng.normal(2_000_000, 300_000, 50)
    volume[250:300] = rng.normal(700_000, 100_000, 50)
    volume[300:380] = rng.normal(1_000_000, 200_000, 80)
    volume[380:420] = rng.normal(800_000, 150_000, 40)
    volume[420:460] = rng.normal(600_000, 100_000, 40)
    volume[460:500] = rng.normal(400_000, 80_000, 40)
    volume[500:550] = rng.normal(2_000_000, 300_000, 50)
    volume[550:] = rng.normal(700_000, 100_000, 50)
    volume = np.clip(volume, 100_000, None)

    close = prices
    high = close * (1 + rng.uniform(0, 0.015, n))
    low = close * (1 - rng.uniform(0, 0.015, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SYN_VCP_LONG",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume.astype(int),
        }
    )


def _make_syn_earnings_gap(rng: np.random.Generator) -> pd.DataFrame:
    """SYN_EARNINGS_GAP: Uptrend with a 7% gap-up at bar 220 (for earnings gap open tests).

    250 bars: steady uptrend (price > SMA200) with a single gap-up event.
    """
    dates = _trading_dates()
    n = len(dates)

    # Steady uptrend: ~0.2% daily drift
    prices = np.zeros(n)
    prices[0] = 100.0
    for i in range(1, n):
        prices[i] = prices[i - 1] * (1 + rng.normal(0.002, 0.005))

    # Insert 7% gap-up at bar 220
    gap_idx = 220
    gap_mult = 1.07
    open_ = prices.copy()
    open_[gap_idx] = prices[gap_idx - 1] * gap_mult
    prices[gap_idx] = open_[gap_idx] * 1.01  # close slightly above open
    # Continue uptrend after gap
    for i in range(gap_idx + 1, n):
        prices[i] = prices[i - 1] * (1 + rng.normal(0.001, 0.005))

    close = prices
    high = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, n))
    low = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, n))
    volume = rng.normal(1_000_000, 200_000, n).astype(int)
    volume = np.clip(volume, 100_000, None)

    return pd.DataFrame(
        {
            "timestamp": dates,
            "symbol": "SYN_EARNINGS_GAP",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _make_sector_fixture() -> pd.DataFrame:
    """10-row sector classification with 3-4 sectors, >30% Tech exposure."""
    return pd.DataFrame(
        {
            "ticker": [
                "SYN_VCP",
                "SYN_VCP_LONG",
                "SYN_TREND",
                "SYN_NOTREND",
                "SYN_GAPUP",
                "SYN_GAPDOWN",
                "SYN_EARNINGS_GAP",
                "AAPL",
                "MSFT",
                "JPM",
                "XOM",
                "JNJ",
            ],
            "sector": [
                "Technology",
                "Technology",
                "Technology",
                "Healthcare",
                "Technology",
                "Energy",
                "Technology",
                "Technology",
                "Technology",
                "Financials",
                "Energy",
                "Healthcare",
            ],
            "industry": [
                "Software",
                "Software",
                "Software",
                "Biotech",
                "Semiconductors",
                "Oil & Gas",
                "Software",
                "Consumer Electronics",
                "Software",
                "Banking",
                "Oil & Gas",
                "Pharmaceuticals",
            ],
            "updated_at": pd.Timestamp("2026-02-20", tz="UTC"),
        }
    )


def generate_all() -> None:
    """Generate all fixture files."""
    rng = np.random.default_rng(SEED)

    # Sample OHLCV: 6 symbols x 250 days + 1 symbol x 600 days
    dfs = [
        _make_syn_vcp(rng),
        _make_syn_vcp_long(rng),
        _make_syn_trend(rng),
        _make_syn_notrend(rng),
        _make_syn_gapup(rng),
        _make_syn_gapdown(rng),
        _make_syn_earnings_gap(rng),
    ]
    ohlcv = pd.concat(dfs, ignore_index=True)
    ohlcv["timestamp"] = ohlcv["timestamp"].dt.tz_localize("UTC")
    ohlcv.to_parquet(FIXTURES_DIR / "sample_ohlcv.parquet", index=False)
    print(f"Generated sample_ohlcv.parquet: {len(ohlcv)} rows")

    # Sample VCP signal: subset for signal testing
    vcp_data = ohlcv[ohlcv["symbol"] == "SYN_VCP"].copy()
    vcp_data.to_parquet(FIXTURES_DIR / "sample_vcp_signal.parquet", index=False)
    print(f"Generated sample_vcp_signal.parquet: {len(vcp_data)} rows")

    # Sector classification
    sectors = _make_sector_fixture()
    sectors.to_parquet(FIXTURES_DIR / "sample_sector.parquet", index=False)
    print(f"Generated sample_sector.parquet: {len(sectors)} rows")


if __name__ == "__main__":
    generate_all()
