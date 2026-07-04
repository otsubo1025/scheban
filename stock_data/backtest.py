"""学習済みモデルを使った全自動ペーパートレードのリプレイ（ステップ3）。

※ 実発注は一切しない、趣味のシミュレーションです。

ルール:
  - 対象: 主要10銘柄（tickers.py）
  - 期間: 各銘柄のテスト期間を1日ずつ順に進めるリプレイ形式
  - 毎日、10銘柄の「翌日上昇確率」を予測し、確率の高い順に 100株ずつ購入
      * 初期資金 100万円、保有は最大3銘柄まで、資金が足りなければ買わない
  - 保有銘柄が下落予測（上昇確率 < 閾値）に転じたら 100株売り
  - 売買手数料あり（約定代金 × fee_rate）
  - 集計: 資産推移・損益・最大ドローダウン・勝率
  - 比較: 等ウェイトの buy & hold
  - 記録: 各取引に「銘柄・上昇確率・SHAP上位要因」を添えた売買ログ

リーク回避:
  - 各銘柄のモデルは学習期間だけで学習（train_model.time_split / train_lgbm）。
  - 日 t の意思決定は「日 t までで確定した特徴量」による予測のみを使い、
    約定はその日の終値で行う（未来価格は使わない）。

使い方:
    python backtest.py                    # 全10銘柄・合成 or 実データでリプレイ
    python backtest.py --max-holdings 3 --fee-rate 0.0005 --threshold 0.5
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import shap

from features import FEATURE_COLUMNS, TARGET_COLUMN, build_dataset
from synthetic_data import load_or_synthesize
from tickers import TICKERS
from train_model import time_split, train_lgbm

OUT_DIR = Path(__file__).resolve().parent / "data"


def _top_shap_string(shap_row: np.ndarray, top_k: int = 3) -> str:
    """1行分の SHAP 値から寄与の大きい特徴量トップ K を文字列化。"""
    pairs = sorted(zip(FEATURE_COLUMNS, shap_row), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return ", ".join(f"{name}({val:+.3f})" for name, val in pairs)


def prepare_symbol(symbol: str, test_size: float) -> tuple[pd.DataFrame, str]:
    """1銘柄を学習し、テスト期間の (Date, Close, up_prob, top_shap) を返す。"""
    raw, source = load_or_synthesize(symbol)
    dataset = build_dataset(raw)
    train, test = time_split(dataset, test_size)
    model = train_lgbm(train)

    proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]

    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        shap_out = explainer.shap_values(test[FEATURE_COLUMNS])
    shap_values = np.asarray(shap_out[1]) if isinstance(shap_out, list) else np.asarray(shap_out)
    top = [_top_shap_string(shap_values[i]) for i in range(len(test))]

    frame = pd.DataFrame(
        {
            "Close": test["Close"].to_numpy(),
            "up_prob": proba,
            "top_shap": top,
        },
        index=pd.to_datetime(test["Date"].to_numpy()),
    )
    frame.index.name = "Date"
    return frame, source


def max_drawdown(equity: pd.Series) -> float:
    """最大ドローダウン（ピークからの最大下落率, 正の値で返す）。"""
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    return float(-drawdown.min())


def run_backtest(
    symbols: list[str],
    test_size: float,
    initial_cash: float,
    shares_per_trade: int,
    max_holdings: int,
    fee_rate: float,
    threshold: float,
) -> dict:
    frames: dict[str, pd.DataFrame] = {}
    source = "synthetic"
    for s in symbols:
        frames[s], source = prepare_symbol(s, test_size)

    # 全銘柄で共通して予測がある営業日だけを対象にする（アラインメント）
    common = None
    for f in frames.values():
        idx = set(f.index)
        common = idx if common is None else (common & idx)
    dates = sorted(common)

    cash = initial_cash
    positions: dict[str, dict] = {}  # symbol -> {shares, entry_price, entry_fee, entry_date}
    trades: list[dict] = []
    equity_rows: list[dict] = []

    def fee_of(value: float) -> float:
        return value * fee_rate

    for d in dates:
        # 1) 売り: 保有銘柄が下落予測に転じたら 100株売り
        for s in list(positions):
            prob = float(frames[s].at[d, "up_prob"])
            if prob < threshold:
                pos = positions[s]
                price = float(frames[s].at[d, "Close"])
                shares = pos["shares"]
                gross = price * shares
                sell_fee = fee_of(gross)
                cash += gross - sell_fee
                realized = (price - pos["entry_price"]) * shares - pos["entry_fee"] - sell_fee
                trades.append(
                    {
                        "date": d.date().isoformat(),
                        "action": "SELL",
                        "symbol": s,
                        "name": TICKERS.get(s, s),
                        "shares": shares,
                        "price": round(price, 2),
                        "up_prob": round(prob, 4),
                        "fee": round(sell_fee, 2),
                        "realized_pnl": round(realized, 2),
                        "cash_after": round(cash, 2),
                        "top_shap": frames[s].at[d, "top_shap"],
                        "reason": "下落予測に転換",
                    }
                )
                del positions[s]

        # 2) 買い: 上昇確率の高い順に、保有していない銘柄を 100株ずつ購入
        candidates = [
            (s, float(frames[s].at[d, "up_prob"]))
            for s in symbols
            if s not in positions and float(frames[s].at[d, "up_prob"]) >= threshold
        ]
        candidates.sort(key=lambda kv: kv[1], reverse=True)
        for s, prob in candidates:
            if len(positions) >= max_holdings:
                break
            price = float(frames[s].at[d, "Close"])
            cost = price * shares_per_trade
            buy_fee = fee_of(cost)
            if cash < cost + buy_fee:  # 資金不足なら買わない
                continue
            cash -= cost + buy_fee
            positions[s] = {
                "shares": shares_per_trade,
                "entry_price": price,
                "entry_fee": buy_fee,
                "entry_date": d,
            }
            trades.append(
                {
                    "date": d.date().isoformat(),
                    "action": "BUY",
                    "symbol": s,
                    "name": TICKERS.get(s, s),
                    "shares": shares_per_trade,
                    "price": round(price, 2),
                    "up_prob": round(prob, 4),
                    "fee": round(buy_fee, 2),
                    "realized_pnl": None,
                    "cash_after": round(cash, 2),
                    "top_shap": frames[s].at[d, "top_shap"],
                    "reason": "上昇確率上位",
                }
            )

        # 3) 時価評価（その日の終値で保有を評価）
        holdings_value = sum(pos["shares"] * float(frames[s].at[d, "Close"]) for s, pos in positions.items())
        equity_rows.append(
            {"date": d.date().isoformat(), "equity": cash + holdings_value, "cash": cash, "n_positions": len(positions)}
        )

    # 最終日リプレイ後・手仕舞い前の「現在の保有状況」スナップショット
    last_d = dates[-1]
    snapshot_cash = cash
    snapshot_rows = []
    for s, pos in positions.items():
        last_price = float(frames[s].at[last_d, "Close"])
        market_value = pos["shares"] * last_price
        snapshot_rows.append(
            {
                "symbol": s,
                "name": TICKERS.get(s, s),
                "shares": pos["shares"],
                "entry_date": pos["entry_date"].date().isoformat(),
                "entry_price": round(pos["entry_price"], 2),
                "last_price": round(last_price, 2),
                "market_value": round(market_value, 2),
                "unrealized_pnl": round((last_price - pos["entry_price"]) * pos["shares"] - pos["entry_fee"], 2),
            }
        )
    positions_df = pd.DataFrame(
        snapshot_rows,
        columns=["symbol", "name", "shares", "entry_date", "entry_price", "last_price", "market_value", "unrealized_pnl"],
    )

    # 最終日にポジションを手仕舞い（実現損益を確定して勝率を集計しやすくする）
    for s in list(positions):
        pos = positions[s]
        price = float(frames[s].at[last_d, "Close"])
        shares = pos["shares"]
        gross = price * shares
        sell_fee = fee_of(gross)
        cash += gross - sell_fee
        realized = (price - pos["entry_price"]) * shares - pos["entry_fee"] - sell_fee
        trades.append(
            {
                "date": last_d.date().isoformat(),
                "action": "SELL",
                "symbol": s,
                "name": TICKERS.get(s, s),
                "shares": shares,
                "price": round(price, 2),
                "up_prob": round(float(frames[s].at[last_d, "up_prob"]), 4),
                "fee": round(sell_fee, 2),
                "realized_pnl": round(realized, 2),
                "cash_after": round(cash, 2),
                "top_shap": frames[s].at[last_d, "top_shap"],
                "reason": "最終日クローズ",
            }
        )
        del positions[s]

    equity_df = pd.DataFrame(equity_rows)
    equity_df.loc[equity_df.index[-1], "equity"] = cash  # 手仕舞い後の確定資産
    equity_series = pd.Series(equity_df["equity"].to_numpy(), index=pd.to_datetime(equity_df["date"]))

    trades_df = pd.DataFrame(trades)

    # 比較用: 等ウェイト buy & hold（初日終値で10銘柄に等分投資し最終日まで保有）
    bh_equity = _buy_and_hold(frames, dates, initial_cash, fee_rate)
    equity_df["buyhold_equity"] = bh_equity.to_numpy()

    # 集計
    closed = trades_df[trades_df["action"] == "SELL"]
    wins = int((closed["realized_pnl"] > 0).sum())
    n_closed = len(closed)
    strat_metrics = {
        "final_equity": float(equity_series.iloc[-1]),
        "total_return": float(equity_series.iloc[-1] / initial_cash - 1),
        "max_drawdown": max_drawdown(equity_series),
        "n_trades": int(len(trades_df)),
        "n_closed": n_closed,
        "win_rate": (wins / n_closed) if n_closed else float("nan"),
        "total_pnl": float(closed["realized_pnl"].sum()) if n_closed else 0.0,
    }
    bh_metrics = {
        "final_equity": float(bh_equity.iloc[-1]),
        "total_return": float(bh_equity.iloc[-1] / initial_cash - 1),
        "max_drawdown": max_drawdown(bh_equity),
    }

    return {
        "source": source,
        "dates": dates,
        "equity_df": equity_df,
        "trades_df": trades_df,
        "positions_df": positions_df,
        "snapshot_cash": float(snapshot_cash),
        "strategy": strat_metrics,
        "buyhold": bh_metrics,
    }


def _buy_and_hold(frames: dict[str, pd.DataFrame], dates: list, initial_cash: float, fee_rate: float) -> pd.Series:
    """等ウェイト buy & hold のエクイティ曲線。"""
    symbols = list(frames)
    alloc = initial_cash / len(symbols)
    units = {}
    for s in symbols:
        first_price = float(frames[s].at[dates[0], "Close"])
        invest = alloc * (1 - fee_rate)  # 初日に一度だけ手数料
        units[s] = invest / first_price
    values = []
    for d in dates:
        v = sum(units[s] * float(frames[s].at[d, "Close"]) for s in symbols)
        values.append(v)
    return pd.Series(values, index=pd.to_datetime([d for d in dates]))


def print_report(res: dict) -> None:
    s, b = res["strategy"], res["buyhold"]
    dates = res["dates"]
    print("=" * 68)
    print(f"ペーパートレード・リプレイ結果   データ種別: {res['source']}")
    print(f"対象: {len(TICKERS)}銘柄  期間: {dates[0].date()} 〜 {dates[-1].date()}  ({len(dates)}営業日)")
    print("-" * 68)
    print(f"{'指標':<20}{'戦略':>20}{'Buy&Hold':>22}")
    print(f"{'最終資産(円)':<20}{s['final_equity']:>20,.0f}{b['final_equity']:>22,.0f}")
    print(f"{'トータルリターン':<18}{s['total_return']:>20.2%}{b['total_return']:>22.2%}")
    print(f"{'最大ドローダウン':<18}{s['max_drawdown']:>20.2%}{b['max_drawdown']:>22.2%}")
    print("-" * 68)
    print(f"取引回数: {s['n_trades']}  (うち決済 {s['n_closed']})   勝率: {s['win_rate']:.2%}   実現損益合計: {s['total_pnl']:,.0f}円")
    print("=" * 68)


def run(args: argparse.Namespace) -> dict:
    symbols = args.symbols if args.symbols else list(TICKERS)
    res = run_backtest(
        symbols=symbols,
        test_size=args.test_size,
        initial_cash=args.initial_cash,
        shares_per_trade=args.shares,
        max_holdings=args.max_holdings,
        fee_rate=args.fee_rate,
        threshold=args.threshold,
    )
    print_report(res)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trades_path = OUT_DIR / "backtest_trades.csv"
    equity_path = OUT_DIR / "backtest_equity.csv"
    positions_path = OUT_DIR / "backtest_positions.csv"
    summary_path = OUT_DIR / "backtest_summary.csv"
    res["trades_df"].to_csv(trades_path, index=False, encoding="utf-8")
    res["equity_df"].to_csv(equity_path, index=False, encoding="utf-8")
    res["positions_df"].to_csv(positions_path, index=False, encoding="utf-8")

    # ダッシュボードが計算せず表示するだけで済むよう、サマリー指標を1行 CSV に保存
    s = res["strategy"]
    dates = res["dates"]
    holdings_value = float(res["positions_df"]["market_value"].sum()) if not res["positions_df"].empty else 0.0
    summary = pd.DataFrame(
        [
            {
                "source": res["source"],
                "start": dates[0].date().isoformat(),
                "end": dates[-1].date().isoformat(),
                "initial_cash": args.initial_cash,
                "final_equity": s["final_equity"],
                "total_return": s["total_return"],
                "max_drawdown": s["max_drawdown"],
                "win_rate": s["win_rate"],
                "total_pnl": s["total_pnl"],
                "n_trades": s["n_trades"],
                "n_closed": s["n_closed"],
                "cash": res["snapshot_cash"],
                "holdings_value": holdings_value,
                "buyhold_return": res["buyhold"]["total_return"],
                "buyhold_max_drawdown": res["buyhold"]["max_drawdown"],
            }
        ]
    )
    summary.to_csv(summary_path, index=False, encoding="utf-8")

    print("\n--- 売買ログ（先頭8件）---")
    cols = ["date", "action", "symbol", "shares", "price", "up_prob", "realized_pnl", "top_shap"]
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(res["trades_df"][cols].head(8).to_string(index=False))
    print(f"\n売買ログ:   {trades_path}")
    print(f"資産推移:   {equity_path}")
    print(f"保有状況:   {positions_path}")
    print(f"サマリー:   {summary_path}")
    return res


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbols", nargs="*", default=None, help="対象銘柄（省略時は主要10銘柄）")
    p.add_argument("--test-size", type=float, default=0.3, help="後半をテスト期間にする割合")
    p.add_argument("--initial-cash", type=float, default=1_000_000, help="初期資金（円）")
    p.add_argument("--shares", type=int, default=100, help="1回の売買株数")
    p.add_argument("--max-holdings", type=int, default=3, help="最大同時保有銘柄数")
    p.add_argument("--fee-rate", type=float, default=0.0005, help="売買手数料率（約定代金比）")
    p.add_argument("--threshold", type=float, default=0.5, help="上昇/下落判定の確率閾値")
    return p.parse_args(argv)


def main(argv=None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
