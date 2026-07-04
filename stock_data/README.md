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

## ステップ2: LightGBM で「翌日上がるか」を予測

取得済みの日足（無ければ合成データ）から特徴量を作り、LightGBM で
「翌日の終値が当日より上がるか（1/0）」を予測します。

```bash
python train_model.py                 # 7203.T（実データ無ければ合成データ）
python train_model.py --symbol 6758.T
python train_model.py --test-size 0.3 # 後半30%をテストに
```

### 設計のポイント

- **特徴量**（`features.py`）
  - 移動平均乖離（5日/25日）、5日線vs25日線、過去リターン（1/5/10日）、
    RSI(14)、ボラティリティ(10日)、出来高変化・出来高比、当日値幅。
- **目的変数**: `close[t+1] > close[t]`（翌日終値が当日より上か）。
- **時系列分割**: 前半で学習 → 後半でテスト。**シャッフルは一切しない**
  （early stopping 用の検証も学習期間の末尾から時系列で取得）。
- **データリーク厳禁**:
  - 特徴量は rolling / shift / pct_change のみで作り「その日まで」の情報だけを使用。
  - 目的変数は未来（t+1）を使うラベルだが特徴量には混ぜない。翌日が無い最終行は除去。
  - この不変条件は `python check_leakage.py` で機械的に検証できる。

### 出力

- テスト期間の **正解率 (Accuracy)** と **AUC**、上昇比率、ベースライン比較。
- 各予測の **上昇確率** と **SHAP** による特徴量寄与度 → `data/predictions_shap_<symbol>.csv`。
- 特徴量ごとの平均寄与度（|SHAP| 平均）。

> 合成データはほぼランダムウォークのため、AUC は 0.5 付近になります（これは正常）。
> リークがあれば不自然に高い精度が出るため、0.5 付近であること自体がリーク無しの傍証です。
> 実データに差し替えると意味のある精度評価ができます。

## ステップ3: 全自動ペーパートレードのリプレイ

> ⚠️ 実発注は一切しない、趣味のシミュレーションです。

学習済みモデルで主要10銘柄の「翌日上昇確率」を毎日予測し、テスト期間を
1日ずつ進めながら自動売買を再現します。

```bash
python backtest.py                                  # 10銘柄・初期資金100万円
python backtest.py --max-holdings 3 --fee-rate 0.0005 --threshold 0.5
```

### ルール

- 毎日、10銘柄の上昇確率を予測し、**確率の高い順に100株ずつ購入**。
- 初期資金 **100万円**、保有は**最大3銘柄**まで、**資金が足りなければ買わない**。
- 保有銘柄が**下落予測（確率 < 閾値）に転じたら100株売り**。
- **売買手数料**あり（約定代金 × `--fee-rate`、既定 0.05%）。
- 各銘柄のモデルは**学習期間のみ**で学習し、意思決定は「その日までの特徴量」に
  よる予測だけを使用（約定はその日の終値／未来価格は不使用）。

### 集計・比較・記録

- **資産推移・トータルリターン・最大ドローダウン・勝率・実現損益**を表示。
- 比較用に**等ウェイト buy & hold** のエクイティ曲線と指標も算出。
- `data/backtest_trades.csv` … 各取引に**銘柄・上昇確率・SHAP上位要因**を添えた売買ログ。
- `data/backtest_equity.csv` … 日次の資産推移（戦略 vs buy & hold）。

> 合成データはランダムウォークのため、戦略が buy & hold に勝つ回も負ける回も
> あります（有意な優劣ではありません）。ここで確認するのは**売買ロジックが
> 未来を先読みせずエンドツーエンドで回ること**です。実データに差し替えて評価してください。

## ステップ4: Streamlit ダッシュボード

バックテスト結果（`data/backtest_*.csv`）を読み込んで表示するダッシュボードです。
**重い計算はアプリ内で行わず、CSV を読んで表示するだけ**（メモリ節約）。

```bash
pip install -r requirements.txt
python backtest.py          # 先に CSV を生成（未生成なら）
streamlit run dashboard.py  # ブラウザで開く
```

表示内容:
- 上部の**サマリー指標**: トータルリターン・最大ドローダウン・勝率・実現損益（Buy&Hold比較つき）
- **資産推移グラフ**（戦略 vs buy & hold）
- **現在の保有状況**（銘柄・株数・評価損益・現金）
- **理由つき売買ログ**（日付・売買・銘柄・株数・価格・上昇確率・SHAP上位要因・理由）
  … 銘柄・期間で絞り込み可能
- スマホ向けに centered レイアウト＋指標は2列で折り返し。

読み込む CSV（すべて `backtest.py` が生成）:
`backtest_summary.csv` / `backtest_equity.csv` / `backtest_positions.csv` / `backtest_trades.csv`

### クラウド（Streamlit Community Cloud）へのデプロイ

1. このリポジトリを GitHub に push。
2. Streamlit Community Cloud で **Main file path** を `stock_data/dashboard.py` に設定。
   依存は同ディレクトリの `stock_data/requirements.txt` から解決されます。
3. リポジトリには**デモ表示用の合成データ `data/backtest_*.csv`** を含めているため、
   デプロイ直後から表示できます（実データで再生成すれば内容が置き換わります）。

## 注意

- ステップ1の実データ取得は Yahoo Finance へアクセスできるネットワーク環境で実行してください。
  egress が制限された環境（社内プロキシ等）では取得に失敗します。
- yfinance は非公式 API のため、Yahoo 側の仕様変更やレート制限の影響を受けることがあります。
  失敗時はリトライしますが、時間をおいて再実行してください。
