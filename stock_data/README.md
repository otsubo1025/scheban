# stock_data — 日本株データ取得（ステップ1）

主要10銘柄（東証）の過去5年分の**日足**を [yfinance](https://pypi.org/project/yfinance/) で取得し、
銘柄ごとに CSV へ保存します。あわせて、ちゃんと取れているか（**行数・期間・欠損・重複**）を
確認できるサマリを出力します。

## 対象銘柄（主要10銘柄）

| symbol | 会社名 |
| --- | --- |
| 7203.T | トヨタ自動車 |
| 6758.T | ソニーグループ |
| 7974.T | 任天堂 |
| 9984.T | ソフトバンクグループ |
| 6861.T | キーエンス |
| 8306.T | 三菱UFJフィナンシャル・グループ |
| 9432.T | 日本電信電話（NTT） |
| 6098.T | リクルートホールディングス |
| 8035.T | 東京エレクトロン |
| 6501.T | 日立製作所 |

銘柄は `tickers.py` で定義しています。

## セットアップ

```bash
cd stock_data
python -m venv .venv && source .venv/bin/activate   # 任意
pip install -r requirements.txt
```

## 実行

```bash
python fetch_stocks.py                    # 主要10銘柄・5年・日足
python fetch_stocks.py --period 3y        # 期間を変更
python fetch_stocks.py --tickers 7203.T 6758.T   # 銘柄を絞る
```

## 出力

- `data/<symbol>.csv` … 銘柄ごとの日足（`Date, Open, High, Low, Close, Volume, Dividends, Stock Splits`。
  終値は `auto_adjust=True` による調整済み）
- `data/summary.csv` … 銘柄ごとの検証結果（行数・期間・欠損・重複・ステータス）

実行の最後に、標準出力へ次のようなサマリが表示されます。

```
=== 取得結果サマリ ===
symbol   name  rows      start        end  duplicate_dates  missing_total status
7203.T トヨタ自動車  1230 2020-07-06 2025-07-04                0              0     OK
...
成功(OK): 10/10 銘柄
```

1銘柄でも取得できなかった場合、終了コードは非ゼロになります。

## 注意

- Yahoo Finance へアクセスできるネットワーク環境で実行してください。
  egress が制限された環境（社内プロキシ等）では取得に失敗します。
- yfinance は非公式 API のため、Yahoo 側の仕様変更やレート制限の影響を受けることがあります。
  失敗時はリトライしますが、時間をおいて再実行してください。
