"""主要10銘柄（日本株）の過去5年分の日足を yfinance で取得し CSV に保存するスクリプト。

ステップ1:
  - 主要10銘柄（トヨタ 7203.T / ソニー 6758.T / 任天堂 7974.T ほか）の日足を取得
  - 銘柄ごとに CSV へ保存（data/<symbol>.csv）
  - 取得できているか（行数・期間・欠損）を確認できるサマリを出力（data/summary.csv）

使い方:
    python fetch_stocks.py                 # 5年・日足で全銘柄取得
    python fetch_stocks.py --period 3y     # 期間を変更
    python fetch_stocks.py --tickers 7203.T 6758.T   # 銘柄を絞る

このスクリプトは Yahoo Finance へアクセスできるネットワーク環境で実行してください。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    sys.exit(
        "yfinance がインストールされていません。`pip install -r requirements.txt` を実行してください。"
    )

from tickers import TICKERS

# 検証対象とする主要な価格・出来高カラム
PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

DATA_DIR = Path(__file__).resolve().parent / "data"


def fetch_one(
    symbol: str,
    period: str,
    interval: str,
    retries: int = 3,
    pause: float = 1.0,
) -> pd.DataFrame:
    """1銘柄分の株価データを取得する。失敗時はリトライする。

    戻り値は Date を列に持つ DataFrame（取得できなければ空の DataFrame）。
    """
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                period=period,
                interval=interval,
                auto_adjust=True,
                actions=True,
            )
            if not df.empty:
                df = df.reset_index()
                # 列名のゆらぎ（Date / Datetime）を吸収
                if "Datetime" in df.columns and "Date" not in df.columns:
                    df = df.rename(columns={"Datetime": "Date"})
                # タイムゾーン付きの場合は日付だけにそろえる
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
                return df
            last_err = RuntimeError("空のデータが返されました")
        except Exception as err:  # ネットワーク等の一時的な失敗
            last_err = err
        if attempt < retries:
            wait = pause * attempt
            print(f"    [{symbol}] 取得失敗（{attempt}/{retries}）: {last_err} -> {wait:.0f}秒待機して再試行")
            time.sleep(wait)
    print(f"    [{symbol}] 取得に失敗しました: {last_err}")
    return pd.DataFrame()


def summarize(symbol: str, name: str, df: pd.DataFrame) -> dict:
    """1銘柄分のデータを検証してサマリ（行数・期間・欠損）を作る。"""
    if df.empty:
        return {
            "symbol": symbol,
            "name": name,
            "rows": 0,
            "start": None,
            "end": None,
            "duplicate_dates": 0,
            "missing_total": None,
            **{f"missing_{col}": None for col in PRICE_COLUMNS},
            "status": "取得失敗",
        }

    present_cols = [c for c in PRICE_COLUMNS if c in df.columns]
    missing_per_col = {f"missing_{c}": int(df[c].isna().sum()) for c in present_cols}
    # 存在しないカラムは None 埋め（表の列を揃えるため）
    for c in PRICE_COLUMNS:
        missing_per_col.setdefault(f"missing_{c}", None)

    missing_total = int(sum(v for v in missing_per_col.values() if v is not None))
    dup = int(df["Date"].duplicated().sum()) if "Date" in df.columns else 0

    status = "OK" if missing_total == 0 and dup == 0 else "要確認"

    return {
        "symbol": symbol,
        "name": name,
        "rows": int(len(df)),
        "start": df["Date"].min().date().isoformat() if "Date" in df.columns else None,
        "end": df["Date"].max().date().isoformat() if "Date" in df.columns else None,
        "duplicate_dates": dup,
        "missing_total": missing_total,
        **missing_per_col,
        "status": status,
    }


def print_summary(summary_df: pd.DataFrame) -> None:
    """サマリを人が読める形で標準出力に表示する。"""
    print("\n=== 取得結果サマリ ===")
    cols = ["symbol", "name", "rows", "start", "end", "duplicate_dates", "missing_total", "status"]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summary_df[cols].to_string(index=False))

    ok = int((summary_df["status"] == "OK").sum())
    total = len(summary_df)
    failed = summary_df[summary_df["rows"] == 0]["symbol"].tolist()
    print(f"\n成功(OK): {ok}/{total} 銘柄")
    if failed:
        print(f"取得失敗: {', '.join(failed)}")
    warn = summary_df[summary_df["status"] == "要確認"]["symbol"].tolist()
    if warn:
        print(f"欠損/重複あり（要確認）: {', '.join(warn)}")


def run(tickers: dict[str, str], period: str, interval: str, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict] = []

    for i, (symbol, name) in enumerate(tickers.items(), start=1):
        print(f"[{i}/{len(tickers)}] {symbol} ({name}) を取得中 ...")
        df = fetch_one(symbol, period=period, interval=interval)
        if not df.empty:
            out = output_dir / f"{symbol}.csv"
            df.to_csv(out, index=False, encoding="utf-8")
            print(f"    -> {len(df)} 行を保存: {out}")
        summaries.append(summarize(symbol, name, df))
        # Yahoo 側のレート制限を避けるため少し待つ
        time.sleep(0.5)

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")
    print_summary(summary_df)
    print(f"\nサマリを保存しました: {summary_path}")
    return summary_df


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="5y", help="取得期間（既定: 5y）")
    parser.add_argument("--interval", default="1d", help="足の間隔（既定: 1d）")
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="取得する銘柄シンボル（省略時は主要10銘柄）",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR),
        help=f"CSV の出力先（既定: {DATA_DIR}）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.tickers:
        tickers = {s: TICKERS.get(s, s) for s in args.tickers}
    else:
        tickers = TICKERS

    summary_df = run(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        output_dir=Path(args.output_dir),
    )
    # 1銘柄でも取得できなければ非ゼロ終了（CI 等で検知しやすくする）
    return 0 if (summary_df["rows"] > 0).all() else 1


if __name__ == "__main__":
    raise SystemExit(main())
