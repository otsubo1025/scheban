"""バックテスト結果を表示する Streamlit ダッシュボード（ステップ4）。

方針:
  - 重い計算はここでは行わず、backtest.py が出力した CSV を読んで表示するだけ。
    （メモリ節約のため。指標も backtest_summary.csv から読むだけ）
  - スマホでも見やすいように centered レイアウト＋指標は少数列で折り返し。
  - ローカルでもクラウド（Streamlit Community Cloud 等）でも動くよう、
    データパスはこのファイルからの相対で解決する。

起動:
    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent / "data"

st.set_page_config(page_title="ペーパートレード・ダッシュボード", page_icon="📈", layout="centered")


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame | None:
    """data/<name> を読み込む（無ければ None）。cache でメモリ・再読込を節約。"""
    path = DATA_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


def yen(value: float) -> str:
    return f"¥{value:,.0f}"


def pct(value: float) -> str:
    return f"{value:.2%}" if pd.notna(value) else "—"


# ---- データ読み込み（表示するだけ）----
summary_df = load_csv("backtest_summary.csv")
equity_df = load_csv("backtest_equity.csv")
trades_df = load_csv("backtest_trades.csv")
positions_df = load_csv("backtest_positions.csv")

st.title("📈 ペーパートレード・ダッシュボード")
st.caption("学習済みモデルによる自動売買リプレイの結果（実発注なしのシミュレーション）")

if summary_df is None or equity_df is None:
    st.warning(
        "バックテスト結果の CSV が見つかりません。先に次を実行してください:\n\n"
        "```\npython backtest.py\n```"
    )
    st.stop()

summary = summary_df.iloc[0]

# ---- サマリー指標 ----
st.subheader("サマリー")
st.caption(f"データ種別: {summary['source']} ／ 期間: {summary['start']} 〜 {summary['end']}")

c1, c2 = st.columns(2)
c1.metric("トータルリターン", pct(summary["total_return"]), help="初期資金比の最終リターン")
c2.metric("最大ドローダウン", pct(summary["max_drawdown"]), help="資産ピークからの最大下落率")
c3, c4 = st.columns(2)
c3.metric("勝率", pct(summary["win_rate"]), help="決済した取引のうち利益が出た割合")
c4.metric("実現損益", yen(summary["total_pnl"]))

with st.expander("Buy & Hold との比較"):
    b1, b2 = st.columns(2)
    b1.metric("戦略リターン", pct(summary["total_return"]))
    b2.metric("Buy&Hold リターン", pct(summary["buyhold_return"]))
    b3, b4 = st.columns(2)
    b3.metric("戦略 最大DD", pct(summary["max_drawdown"]))
    b4.metric("Buy&Hold 最大DD", pct(summary["buyhold_max_drawdown"]))

# ---- 資産推移グラフ（戦略 vs buy&hold）----
st.subheader("資産推移")
chart = equity_df.copy()
chart["date"] = pd.to_datetime(chart["date"])
chart = chart.set_index("date")[["equity", "buyhold_equity"]]
chart = chart.rename(columns={"equity": "戦略", "buyhold_equity": "Buy&Hold"})
st.line_chart(chart, height=300)

# ---- 現在の保有状況 ----
st.subheader("現在の保有状況")
h1, h2, h3 = st.columns(3)
h1.metric("現金", yen(summary["cash"]))
h2.metric("評価額", yen(summary["holdings_value"]))
h3.metric("総資産", yen(summary["final_equity"] if positions_df is None or positions_df.empty else summary["cash"] + summary["holdings_value"]))

if positions_df is not None and not positions_df.empty:
    total_unreal = positions_df["unrealized_pnl"].sum()
    st.caption(f"含み損益合計: {yen(total_unreal)}")
    st.dataframe(
        positions_df.rename(
            columns={
                "symbol": "銘柄",
                "name": "名称",
                "shares": "株数",
                "entry_date": "取得日",
                "entry_price": "取得単価",
                "last_price": "現在値",
                "market_value": "評価額",
                "unrealized_pnl": "評価損益",
            }
        ),
        width="stretch",
        hide_index=True,
    )
else:
    st.info("現在の保有はありません（すべて現金）。")

# ---- 理由つき売買ログ ----
st.subheader("売買ログ")
if trades_df is None or trades_df.empty:
    st.info("売買はありませんでした。")
else:
    log = trades_df.copy()
    symbols = sorted(log["symbol"].unique())
    sel_symbols = st.multiselect("銘柄で絞り込み", symbols, default=symbols)

    log["date"] = pd.to_datetime(log["date"])
    dmin, dmax = log["date"].min().date(), log["date"].max().date()
    date_range = st.date_input("期間で絞り込み", value=(dmin, dmax), min_value=dmin, max_value=dmax)

    filtered = log[log["symbol"].isin(sel_symbols)]
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start, end = date_range
        filtered = filtered[(filtered["date"].dt.date >= start) & (filtered["date"].dt.date <= end)]

    st.caption(f"{len(filtered)} 件")
    show_cols = ["date", "action", "symbol", "name", "shares", "price", "up_prob", "realized_pnl", "top_shap", "reason"]
    view = filtered[show_cols].rename(
        columns={
            "date": "日付",
            "action": "売買",
            "symbol": "銘柄",
            "name": "名称",
            "shares": "株数",
            "price": "価格",
            "up_prob": "上昇確率",
            "realized_pnl": "実現損益",
            "top_shap": "SHAP上位要因",
            "reason": "理由",
        }
    )
    view = view.copy()
    view["日付"] = view["日付"].dt.date
    st.dataframe(view, width="stretch", hide_index=True)

st.caption("※ 数値は backtest.py が生成した CSV をそのまま表示しています（本アプリ内で再計算はしません）。")
