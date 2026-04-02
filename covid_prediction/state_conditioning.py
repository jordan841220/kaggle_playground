from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def split_state_and_day_features(
    all_cols: Sequence[str],
    state_count: int,
    day1_pos_col: str,
    baseline_col: str,
) -> tuple[list[str], list[str]]:
    """Split the fixed layout into the state block and the repeated day block.

    The first ``state_count`` columns belong to the one-hot state block. The
    remaining columns are day-window features, except for the two forced scalar
    columns that we always keep outside feature selection.
    """
    state_cols = list(all_cols[:state_count])
    day_feature_cols = [
        col
        for col in all_cols[state_count:]
        if col not in {day1_pos_col, baseline_col}
    ]
    return state_cols, day_feature_cols


def extract_state_indices(df: Any, state_cols: Sequence[str]) -> np.ndarray:
    """Convert a one-hot state block into integer state ids."""
    state_matrix = df[state_cols].to_numpy(dtype=np.float32)
    return np.argmax(state_matrix, axis=1).astype(np.int64)


def build_state_conditioned_inputs(
    X_train: Any,
    X_val: Any,
    y_train: Any,
    state_cols: Sequence[str],
    day_feature_cols: Sequence[str],
    day1_pos_col: str,
    baseline_col: str,
    k: int,
) -> Dict[str, Any]:
    """Build the day-window features and the separate state ids.

    The state block is no longer part of SelectKBest. We keep it as a separate
    integer input so the neural head can condition on a learned embedding.
    """
    candidate_day_cols = list(day_feature_cols)
    k = min(k, len(candidate_day_cols))

    selector = SelectKBest(score_func=f_regression, k=k)
    selector.fit(X_train[candidate_day_cols], y_train)
    selected_feature_cols = [
        col for col, keep in zip(candidate_day_cols, selector.get_support()) if keep
    ]

    day_scaler = StandardScaler()
    day_train_s = day_scaler.fit_transform(X_train[selected_feature_cols]).astype(np.float32)
    day_val_s = day_scaler.transform(X_val[selected_feature_cols]).astype(np.float32)

    day1_pos_scaler = StandardScaler()
    day1_pos_train_raw = X_train[[day1_pos_col]].to_numpy(dtype=np.float32)
    day1_pos_val_raw = X_val[[day1_pos_col]].to_numpy(dtype=np.float32)
    day1_pos_train_s = day1_pos_scaler.fit_transform(day1_pos_train_raw).astype(np.float32)
    day1_pos_val_s = day1_pos_scaler.transform(day1_pos_val_raw).astype(np.float32)

    baseline_scaler = StandardScaler()
    base_train_raw = X_train[[baseline_col]].to_numpy(dtype=np.float32)
    base_val_raw = X_val[[baseline_col]].to_numpy(dtype=np.float32)
    base_train_s = baseline_scaler.fit_transform(base_train_raw).astype(np.float32)
    base_val_s = baseline_scaler.transform(base_val_raw).astype(np.float32)

    X_train_day_s = np.hstack([
        day_train_s,
        day1_pos_train_s,
        base_train_s,
    ])
    X_val_day_s = np.hstack([
        day_val_s,
        day1_pos_val_s,
        base_val_s,
    ])

    state_train_idx = extract_state_indices(X_train, state_cols)
    state_val_idx = extract_state_indices(X_val, state_cols)

    return {
        "selected_feature_cols": selected_feature_cols,
        "day_scaler": day_scaler,
        "feature_scaler": day_scaler,
        "day1_pos_scaler": day1_pos_scaler,
        "baseline_scaler": baseline_scaler,
        "state_train_idx": state_train_idx,
        "state_val_idx": state_val_idx,
        "X_train_day_s": X_train_day_s,
        "X_val_day_s": X_val_day_s,
        "X_train_final_s": X_train_day_s,
        "X_val_final_s": X_val_day_s,
    }


def build_ridge_oof_predictions(
    X_train_s: np.ndarray,
    y_train_np: np.ndarray,
    alpha: float,
    n_splits: int = 5,
    seed: int = 2025,
) -> tuple[np.ndarray, Ridge]:
    """Build out-of-fold Ridge predictions for residual stacking."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_pred = np.zeros((X_train_s.shape[0], 1), dtype=np.float32)

    for fold_tr_idx, fold_oof_idx in kf.split(X_train_s):
        fold_model = Ridge(alpha=alpha)
        fold_model.fit(X_train_s[fold_tr_idx], y_train_np[fold_tr_idx].ravel())
        fold_pred = fold_model.predict(X_train_s[fold_oof_idx]).astype(np.float32).reshape(-1, 1)
        oof_pred[fold_oof_idx] = fold_pred

    full_model = Ridge(alpha=alpha)
    full_model.fit(X_train_s, y_train_np.ravel())
    return oof_pred, full_model


class FiLMBlock(nn.Module):
    """A small MLP block with affine conditioning from the state embedding."""

    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        h = self.linear(x)
        h = self.norm(h)
        h = h * (1.0 + gamma) + beta
        h = self.act(h)
        return self.dropout(h)


class StateConditionedResidualMLP(nn.Module):
    """Predict the ridge residual with a learned state prior and FiLM blocks."""

    def __init__(
        self,
        num_states: int,
        day_input_dim: int,
        state_emb_dim: int = 8,
        hidden: int = 64,
        dropout: float = 0.2,
        n_layers: int = 2,
    ):
        super().__init__()
        self.state_embedding = nn.Embedding(num_states, state_emb_dim)
        self.state_bias = nn.Linear(state_emb_dim, 1)

        self.blocks = nn.ModuleList()
        self.gamma_layers = nn.ModuleList()
        self.beta_layers = nn.ModuleList()

        dim = day_input_dim
        for _ in range(max(1, n_layers)):
            self.blocks.append(FiLMBlock(dim, hidden, dropout))
            self.gamma_layers.append(nn.Linear(state_emb_dim, hidden))
            self.beta_layers.append(nn.Linear(state_emb_dim, hidden))
            dim = hidden

        self.residual_head = nn.Linear(dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.zeros_(self.state_bias.weight)
        nn.init.zeros_(self.state_bias.bias)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, state_idx: torch.Tensor, day_x: torch.Tensor) -> torch.Tensor:
        state_emb = self.state_embedding(state_idx)
        state_bias = self.state_bias(state_emb)

        h = day_x
        for block, gamma_layer, beta_layer in zip(self.blocks, self.gamma_layers, self.beta_layers):
            gamma = 0.1 * torch.tanh(gamma_layer(state_emb))
            beta = beta_layer(state_emb)
            h = block(h, gamma, beta)

        residual = self.residual_head(h)
        return state_bias + residual
