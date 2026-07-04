"""LightGBM で「翌日の終値が上がるか」を予測する（ステップ2）。

方針:
  - まず1銘柄で動かす（--symbol で指定、既定 7203.T）
  - 実データ CSV が data/ に無ければ合成データで動かす
  - 時系列で分割（前半で学習→後半でテスト）。シャッフルは絶対にしない
  - データリーク厳禁（features.py 参照）
  - 出力: テスト期間の正解率・AUC などの精度指標
  - 各予測の上昇確率と、SHAP による特徴量の寄与度を出力

使い方:
    python train_model.py                      # 7203.T（実データ無ければ合成）
    python train_model.py --symbol 6758.T
    python train_model.py --test-size 0.3      # 後半30%をテストに
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

import lightgbm as lgb

from features import FEATURE_COLUMNS, TARGET_COLUMN, build_dataset
from synthetic_data import load_or_synthesize

OUT_DIR = Path(__file__).resolve().parent / "data"


def time_split(dataset: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """時系列で前半＝学習、後半＝テストに分割（シャッフルしない）。"""
    n = len(dataset)
    n_test = int(round(n * test_size))
    n_train = n - n_test
    train = dataset.iloc[:n_train].copy()
    test = dataset.iloc[n_train:].copy()
    return train, test


def train_lgbm(
    train: pd.DataFrame,
    valid_size: float = 0.2,
) -> lgb.LGBMClassifier:
    """LightGBM を学習。early stopping 用の検証も学習期間の“末尾”から時系列で取る。"""
    n = len(train)
    n_valid = int(round(n * valid_size))
    tr = train.iloc[: n - n_valid]
    va = train.iloc[n - n_valid :]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        min_child_samples=30,
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(
        tr[FEATURE_COLUMNS],
        tr[TARGET_COLUMN].astype(int),
        eval_set=[(va[FEATURE_COLUMNS], va[TARGET_COLUMN].astype(int))],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def evaluate(model: lgb.LGBMClassifier, test: pd.DataFrame) -> dict:
    """テスト期間の精度指標を計算する。"""
    y_true = test[TARGET_COLUMN].astype(int).to_numpy()
    proba = model.predict_proba(test[FEATURE_COLUMNS])[:, 1]
    pred = (proba >= 0.5).astype(int)

    metrics = {
        "n_test": len(test),
        "accuracy": accuracy_score(y_true, pred),
        "auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else float("nan"),
        "up_rate_actual": float(y_true.mean()),
        "up_rate_pred": float(pred.mean()),
        # 「常に上昇と予測」した場合のベースライン正解率（比較用）
        "baseline_always_up": float(y_true.mean()),
    }
    return metrics, proba, pred


def shap_report(model: lgb.LGBMClassifier, test: pd.DataFrame, proba: np.ndarray, top_k: int = 3) -> pd.DataFrame:
    """各予測の上昇確率と SHAP 寄与度を DataFrame にまとめる。"""
    import warnings

    import shap

    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():
        # LightGBM 二値分類の出力形式に関する既知の警告（両形式に対応済み）を抑制
        warnings.simplefilter("ignore", UserWarning)
        shap_out = explainer.shap_values(test[FEATURE_COLUMNS])
    # LightGBM 二値分類では (n, features) を返す版と [class0, class1] のリスト版がある
    if isinstance(shap_out, list):
        shap_values = np.asarray(shap_out[1])
    else:
        shap_values = np.asarray(shap_out)

    shap_df = pd.DataFrame(shap_values, columns=[f"shap_{c}" for c in FEATURE_COLUMNS], index=test.index)

    result = pd.DataFrame(
        {
            "Date": test["Date"].values,
            "Close": test["Close"].values,
            "up_probability": proba,
            "actual_up": test[TARGET_COLUMN].astype(int).values,
        },
        index=test.index,
    )
    result = pd.concat([result, shap_df], axis=1)

    # 各行で寄与の大きい特徴量トップ K を文字列でも付ける（人が読む用）
    def top_features(row):
        contribs = {c: row[f"shap_{c}"] for c in FEATURE_COLUMNS}
        ordered = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        return ", ".join(f"{name}({val:+.3f})" for name, val in ordered)

    result["top_shap_features"] = result.apply(top_features, axis=1)
    return result


def print_metrics(symbol: str, source: str, dataset: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, metrics: dict) -> None:
    print("=" * 64)
    print(f"銘柄: {symbol}   データ種別: {source}")
    print(f"総サンプル数: {len(dataset)}（特徴量欠損行は除外済み）")
    print(f"学習期間: {train['Date'].min().date()} 〜 {train['Date'].max().date()}  ({len(train)}件)")
    print(f"テスト期間: {test['Date'].min().date()} 〜 {test['Date'].max().date()}  ({len(test)}件)")
    print("-" * 64)
    print(f"正解率 (Accuracy) : {metrics['accuracy']:.4f}")
    print(f"AUC              : {metrics['auc']:.4f}")
    print(f"実際の上昇比率    : {metrics['up_rate_actual']:.4f}")
    print(f"予測の上昇比率    : {metrics['up_rate_pred']:.4f}")
    print(f"ベースライン(常に上昇と予測の正解率): {metrics['baseline_always_up']:.4f}")
    print("=" * 64)


def run(symbol: str, test_size: float, output_dir: Path) -> dict:
    raw, source = load_or_synthesize(symbol)
    dataset = build_dataset(raw)

    if len(dataset) < 100:
        raise SystemExit("学習に十分なデータがありません。")

    train, test = time_split(dataset, test_size)
    model = train_lgbm(train)
    metrics, proba, pred = evaluate(model, test)
    print_metrics(symbol, source, dataset, train, test, metrics)

    report = shap_report(model, test, proba)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / f"predictions_shap_{symbol}.csv"
    report.to_csv(out_csv, index=False, encoding="utf-8")

    print("\n--- 予測サンプル（テスト期間の末尾5営業日）---")
    cols = ["Date", "Close", "up_probability", "actual_up", "top_shap_features"]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(report[cols].tail(5).to_string(index=False))

    # 特徴量全体の平均的な重要度（|SHAP| の平均）
    shap_cols = [f"shap_{c}" for c in FEATURE_COLUMNS]
    mean_abs = report[shap_cols].abs().mean().sort_values(ascending=False)
    print("\n--- 特徴量の平均寄与度（|SHAP| 平均, 大きい順）---")
    for name, val in mean_abs.items():
        print(f"  {name.replace('shap_',''):18s} {val:.4f}")

    print(f"\n予測＋SHAP を保存しました: {out_csv}")
    return metrics


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--symbol", default="7203.T", help="対象銘柄（既定: 7203.T）")
    p.add_argument("--test-size", type=float, default=0.3, help="後半をテストにする割合（既定: 0.3）")
    p.add_argument("--output-dir", default=str(OUT_DIR), help="出力先")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run(args.symbol, args.test_size, Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
