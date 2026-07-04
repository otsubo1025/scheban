"""実データが無い場合に使う合成の日足データ生成。

`load_or_synthesize()` は data/<symbol>.csv があればそれを読み、無ければ
再現性のある擬似的な株価（幾何ブラウン運動＋緩やかなトレンド）を生成する。
これにより、実データ取得ができない環境でもパイプラインをエンドツーエンドで動かせる。
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"


def synthesize(symbol: str = "7203.T", n_days: int = 1250, seed: int = 42) -> pd.DataFrame:
    """営業日ベースの擬似 OHLCV 日足を生成する（再現性のため seed 固定）。

    返り値の列: Date, Open, High, Low, Close, Volume（yfinance 出力と同じ形）。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end="2025-06-30", periods=n_days)

    # 日次リターン: わずかな上昇トレンド + ノイズ。ボラティリティも緩やかに変動させる。
    drift = 0.0003
    vol = 0.012 * (1 + 0.3 * np.sin(np.linspace(0, 6 * np.pi, n_days)))
    daily_ret = drift + vol * rng.standard_normal(n_days)

    close = 2000 * np.exp(np.cumsum(daily_ret))
    # 当日始値は前日終値付近、高値・安値はその日のレンジ
    prev_close = np.concatenate([[close[0]], close[:-1]])
    open_ = prev_close * (1 + 0.003 * rng.standard_normal(n_days))
    intraday = np.abs(0.008 * rng.standard_normal(n_days)) + 0.002
    high = np.maximum(open_, close) * (1 + intraday)
    low = np.minimum(open_, close) * (1 - intraday)

    # 出来高: 値動きが大きい日ほど増える傾向 + ノイズ
    base_vol = 8_000_000
    volume = (base_vol * (1 + 5 * np.abs(daily_ret)) * (0.5 + rng.random(n_days))).astype(np.int64)

    return pd.DataFrame(
        {
            "Date": dates,
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def load_or_synthesize(symbol: str = "7203.T", data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, str]:
    """実データ CSV があれば読み込み、無ければ合成データを返す。

    戻り値: (DataFrame, source)  source は "real" または "synthetic"。
    """
    csv = data_dir / f"{symbol}.csv"
    if csv.exists():
        df = pd.read_csv(csv, parse_dates=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        return df, "real"
    # 銘柄ごとに違う系列になるよう、シンボルから安定した seed を作る
    # （crc32 は実行間で安定し、char 合計のような衝突が起きにくい）
    seed = zlib.crc32(symbol.encode("utf-8")) % (2**31)
    return synthesize(symbol, seed=seed), "synthetic"
