"""特徴量エンジニアリングと目的変数の作成。

■ データリーク厳禁の方針
  - 特徴量（X）は「その日（t）までに確定している情報」だけで作る。
    rolling / shift / pct_change はいずれも過去〜当日の値のみを参照するため、
    未来（t+1 以降）を先読みしない。
  - 目的変数（y）は「翌日（t+1）の終値が当日（t）より上がるか」= close[t+1] > close[t]。
    これは close.shift(-1) を使うため未来を参照する“ラベル”だが、特徴量には一切混ぜない。
    未来が無い最終行は y が NaN になるので必ず落とす（build_dataset で除去）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# モデルに渡す特徴量カラム（このリストが学習・SHAP の対象）
FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "ma5_dev",
    "ma25_dev",
    "ma5_over_ma25",
    "rsi14",
    "volatility_10d",
    "volume_change_1d",
    "volume_ratio_5d",
    "high_low_range",
]

TARGET_COLUMN = "target_up"


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI（Wilder の単純移動平均版）。過去〜当日の値のみを使用。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    # avg_loss==0（下落なし）の局面は RSI=100 とみなす
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """OHLCV から特徴量を付与する。すべて“その日まで”の情報のみで計算。"""
    out = df.sort_values("Date").reset_index(drop=True).copy()
    close = out["Close"]
    volume = out["Volume"]

    # 過去リターン（当日 t までで確定）
    out["return_1d"] = close.pct_change(1)
    out["return_5d"] = close.pct_change(5)
    out["return_10d"] = close.pct_change(10)

    # 移動平均と乖離率（当日を含む過去 window 日の平均）
    ma5 = close.rolling(5).mean()
    ma25 = close.rolling(25).mean()
    out["ma5_dev"] = close / ma5 - 1.0
    out["ma25_dev"] = close / ma25 - 1.0
    out["ma5_over_ma25"] = ma5 / ma25 - 1.0

    # RSI
    out["rsi14"] = _rsi(close, 14)

    # ボラティリティ（過去10日の日次リターン標準偏差）
    out["volatility_10d"] = out["return_1d"].rolling(10).std()

    # 出来高の変化
    out["volume_change_1d"] = volume.pct_change(1)
    out["volume_ratio_5d"] = volume / volume.rolling(5).mean() - 1.0

    # 当日の値幅（高値-安値）/終値
    out["high_low_range"] = (out["High"] - out["Low"]) / close

    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """目的変数: 翌日終値が当日より上がるか（1/0）。

    close.shift(-1) は未来を参照するラベルなので特徴量とは分離。
    最終行は翌日が無いため NaN（後段で除去）。
    """
    out = df.copy()
    next_close = out["Close"].shift(-1)
    out[TARGET_COLUMN] = (next_close > out["Close"]).astype("float")
    # 翌日が存在しない最終行は未定義にする
    out.loc[out.index[-1], TARGET_COLUMN] = np.nan
    return out


def build_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """特徴量＋目的変数を付け、学習に使える行（欠損なし）だけ残す。"""
    feat = add_features(df)
    feat = add_target(feat)
    needed = ["Date", "Close", *FEATURE_COLUMNS, TARGET_COLUMN]
    clean = feat[needed].dropna().reset_index(drop=True)
    return clean
