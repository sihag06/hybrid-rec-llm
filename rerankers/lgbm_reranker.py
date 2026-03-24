from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd


def _build_group(df: pd.DataFrame, qid_col: str = "qid") -> List[int]:
    return df.groupby(qid_col).size().tolist()


class LGBMReranker:
    """LightGBM LambdaRank-based reranker."""

    def __init__(self, model_path: str, feature_cols: List[str]) -> None:
        self.model_path = model_path
        self.model = lgb.Booster(model_file=model_path)
        self.feature_cols = feature_cols

    def score_batch(self, features: pd.DataFrame) -> np.ndarray:
        feats = features[self.feature_cols].fillna(0.0)
        return self.model.predict(feats)

    def rerank(self, df_candidates: pd.DataFrame, k: int = 10) -> List[str]:
        scores = self.score_batch(df_candidates)
        df_candidates = df_candidates.copy()
        df_candidates["score"] = scores
        df_candidates = df_candidates.sort_values("score", ascending=False)
        return df_candidates["assessment_id"].tolist()[:k]


def train_lgbm_ranker(train_path: str, val_path: str, out_dir: str, feature_cols: List[str]) -> Dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    # Sort by qid for group alignment
    train_df = train_df.sort_values("qid").reset_index(drop=True)
    val_df = val_df.sort_values("qid").reset_index(drop=True)

    X_train = train_df[feature_cols].fillna(0.0)
    y_train = train_df["label"].astype(int).values
    group_train = _build_group(train_df)

    X_val = val_df[feature_cols].fillna(0.0)
    y_val = val_df["label"].astype(int).values
    group_val = _build_group(val_df)

    lgb_train = lgb.Dataset(X_train, label=y_train, group=group_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, group=group_val)

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [10],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }

    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=True)]

    evals_result: Dict[str, Dict[str, List[float]]] = {}
    model = lgb.train(
        params=params,
        train_set=lgb_train,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        num_boost_round=2000,
        callbacks=callbacks + [lgb.record_evaluation(evals_result)],
    )

    model.save_model(out / "lgbm_model.txt")
    with open(out / "feature_schema.json", "w") as f:
        json.dump({"features": feature_cols}, f, indent=2)
    with open(out / "config.json", "w") as f:
        json.dump(params, f, indent=2)
    with open(out / "train_report.json", "w") as f:
        json.dump(evals_result, f, indent=2)

    return {
        "model_path": str(out / "lgbm_model.txt"),
        "best_iteration": model.best_iteration,
        "features": feature_cols,
        "n_train_rows": int(len(train_df)),
        "n_val_rows": int(len(val_df)),
        "n_train_queries": int(train_df["qid"].nunique()),
        "n_val_queries": int(val_df["qid"].nunique()),
    }
