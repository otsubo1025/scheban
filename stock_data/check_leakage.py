"""データリーク（未来情報の混入）が無いことを機械的に検証するチェック。

考え方:
  もし特徴量が“未来”を参照していれば、データを途中で打ち切った時に
  過去行の特徴量が変わってしまう。そこで
    - 全期間で計算した特徴量
    - 先頭 k 行だけに打ち切って計算した特徴量
  を比べ、共通する過去行で完全一致することを確認する（=未来非参照）。

  さらに、目的変数が「翌日終値>当日終値」で正しく作られ、最終行が NaN で
  除去されていることも確認する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS, TARGET_COLUMN, add_features, add_target
from synthetic_data import synthesize


def check_features_no_future(df: pd.DataFrame, cut: int = 800) -> None:
    full = add_features(df)
    truncated = add_features(df.iloc[:cut])
    # 打ち切り前の過去行のみ比較（rolling の性質上、末尾数行の NaN 差は無関係）
    a = full.loc[: cut - 1, FEATURE_COLUMNS].reset_index(drop=True)
    b = truncated.loc[: cut - 1, FEATURE_COLUMNS].reset_index(drop=True)
    # NaN 同士も一致とみなして厳密比較
    mismatch = ~((a == b) | (a.isna() & b.isna()))
    n_bad = int(mismatch.to_numpy().sum())
    assert n_bad == 0, f"特徴量が未来を参照している疑い: {n_bad} セル不一致"
    print(f"[OK] 特徴量は未来を参照していない（先頭{cut}行を全期間計算と照合, 不一致0）")


def check_target_definition(df: pd.DataFrame) -> None:
    t = add_target(add_features(df))
    close = t["Close"].to_numpy()
    y = t[TARGET_COLUMN].to_numpy()
    # 最終行は NaN（翌日が無い）
    assert np.isnan(y[-1]), "最終行の目的変数は NaN であるべき"
    # それ以外は close[t+1] > close[t] と一致
    expected = (close[1:] > close[:-1]).astype(float)
    got = y[:-1]
    assert np.array_equal(got, expected), "目的変数が『翌日終値>当日終値』と一致しない"
    print("[OK] 目的変数は close[t+1] > close[t]。最終行は NaN で除去対象")


def main() -> int:
    df = synthesize(n_days=1000, seed=7)
    check_features_no_future(df, cut=800)
    check_target_definition(df)
    print("\nすべてのリークチェックに合格しました。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
