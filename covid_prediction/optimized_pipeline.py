"""Structured COVID prediction pipeline.

This script is meant to replace the current flat feature-selection MLP with a
more structured approach:

* keep the 40-state one-hot block untouched
* use an explicit Ridge baseline on the full tabular view
* train a residual MLP on top of the Ridge prediction
* model the three day blocks with a shared encoder instead of a flat MLP
* use a state-wise time split for validation so we do not leak future days

The code is written to be runnable from a notebook or as a standalone script.
It expects the duplicate column layout from the Kaggle CSVs.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "covid.train.csv"
TEST_PATH = DATA_DIR / "covid.test.csv"
SUBMISSION_PATH = Path("submission.csv")

STATE_COUNT = 40
SURVEY_COUNT = 17

SEED = 2025
RIDGE_ALPHA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
BLEND_GRID = np.linspace(0.0, 1.0, 21)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def state_ids(df: pd.DataFrame) -> np.ndarray:
    """Return the state index for each row from the first 40 one-hot columns."""
    return df.iloc[:, 1 : 1 + STATE_COUNT].to_numpy(dtype=np.float32).argmax(axis=1)


def split_state_timewise(
    df: pd.DataFrame,
    val_frac: float = 0.2,
    min_val_per_state: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the last chunk of each state block as validation.

    The raw CSV is grouped by state and within each state it is chronological.
    That means a random split leaks future days into the training set.  This
    split keeps the temporal order intact.
    """
    sids = state_ids(df)
    tr_idx: List[int] = []
    va_idx: List[int] = []
    for s in range(STATE_COUNT):
        idx = np.flatnonzero(sids == s)
        if len(idx) == 0:
            continue
        val_count = max(min_val_per_state, int(round(len(idx) * val_frac)))
        val_count = min(val_count, len(idx) - 1)
        tr_idx.extend(idx[:-val_count].tolist())
        va_idx.extend(idx[-val_count:].tolist())
    return df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()


def blocked_state_folds(state_array: np.ndarray, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create blocked folds within each state block.

    Each fold contains contiguous time chunks from every state.  This is a
    lightweight way to generate OOF predictions without shuffling time order.
    """
    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    chunks_by_state = {}
    for s in range(STATE_COUNT):
        idx = np.flatnonzero(state_array == s)
        chunks_by_state[s] = np.array_split(idx, n_splits)

    for fold in range(n_splits):
        train_parts = []
        val_parts = []
        for s in range(STATE_COUNT):
            chunks = chunks_by_state[s]
            val_parts.append(chunks[fold])
            train_parts.extend([chunks[j] for j in range(n_splits) if j != fold and len(chunks[j]) > 0])
        train_idx = np.concatenate(train_parts) if train_parts else np.array([], dtype=int)
        val_idx = np.concatenate(val_parts) if val_parts else np.array([], dtype=int)
        folds.append((train_idx, val_idx))
    return folds


def build_linear_view(
    df: pd.DataFrame,
    scaler: StandardScaler | None = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """Build the wide Ridge view.

    Features:
    * 40 state one-hot columns
    * day1/day2 tested_positive
    * all 17 survey features from day1/day2/day3
    * day2-day1 and day3-day2 deltas for the survey features
    """
    state = df.iloc[:, 1 : 1 + STATE_COUNT].to_numpy(dtype=np.float32)

    day1_survey = df.iloc[:, 41:58].to_numpy(dtype=np.float32)
    day2_survey = df.iloc[:, 59:76].to_numpy(dtype=np.float32)
    day3_survey = df.iloc[:, 77:94].to_numpy(dtype=np.float32)

    day1_tp = df.iloc[:, 58].to_numpy(dtype=np.float32).reshape(-1, 1)
    day2_tp = df.iloc[:, 76].to_numpy(dtype=np.float32).reshape(-1, 1)

    delta21 = day2_survey - day1_survey
    delta32 = day3_survey - day2_survey

    numeric = np.hstack(
        [
            day1_tp,
            day2_tp,
            day1_survey,
            day2_survey,
            day3_survey,
            delta21,
            delta32,
        ]
    ).astype(np.float32)

    if scaler is None:
        scaler = StandardScaler()
        numeric = scaler.fit_transform(numeric).astype(np.float32)
    else:
        numeric = scaler.transform(numeric).astype(np.float32)

    X = np.hstack([state, numeric]).astype(np.float32)
    return X, scaler


def fit_ridge_oof(
    X: np.ndarray,
    y: np.ndarray,
    state_array: np.ndarray,
    alpha: float,
    n_splits: int = 5,
) -> np.ndarray:
    """Blocked OOF predictions for the Ridge baseline."""
    oof = np.zeros(len(y), dtype=np.float32)
    folds = blocked_state_folds(state_array, n_splits=n_splits)
    for tr_idx, va_idx in folds:
        model = Ridge(alpha=alpha)
        model.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict(X[va_idx]).astype(np.float32)
    return oof


@dataclass
class TokenScalers:
    survey_scaler: StandardScaler
    ridge_scaler: StandardScaler
    tp_scaler: StandardScaler


def fit_token_scalers(df: pd.DataFrame, ridge_pred: np.ndarray) -> TokenScalers:
    """Fit scalers for the structured MLP inputs.

    Survey features are standardized across all three day blocks using a shared
    scaler.  The scalar scalers are used for the explicit Ridge prediction and
    for the tested_positive inputs.
    """
    day1_survey = df.iloc[:, 41:58].to_numpy(dtype=np.float32)
    day2_survey = df.iloc[:, 59:76].to_numpy(dtype=np.float32)
    day3_survey = df.iloc[:, 77:94].to_numpy(dtype=np.float32)
    survey_stack = np.vstack([day1_survey, day2_survey, day3_survey]).astype(np.float32)

    survey_scaler = StandardScaler()
    survey_scaler.fit(survey_stack)

    day2_tp = df.iloc[:, 76].to_numpy(dtype=np.float32).reshape(-1, 1)
    ridge_scaler = StandardScaler()
    ridge_scaler.fit(ridge_pred.reshape(-1, 1).astype(np.float32))

    tp_stack = np.vstack([df.iloc[:, 58].to_numpy(dtype=np.float32).reshape(-1, 1), day2_tp]).astype(np.float32)
    tp_scaler = StandardScaler()
    tp_scaler.fit(tp_stack)

    return TokenScalers(survey_scaler=survey_scaler, ridge_scaler=ridge_scaler, tp_scaler=tp_scaler)


def build_token_tensors(
    df: pd.DataFrame,
    ridge_pred: np.ndarray,
    token_scalers: TokenScalers,
) -> Tuple[torch.Tensor, ...]:
    """Convert a dataframe to structured day tokens and scalar tensors.

    Each day token has:
    * 17 standardized survey features
    * standardized tested_positive value if available
    * a mask bit for the tested_positive value
    * a normalized day index in [0, 1]

    Day 3 has no tested_positive input, so its mask is 0 and its scalar is 0.
    """
    state = df.iloc[:, 1 : 1 + STATE_COUNT].to_numpy(dtype=np.float32)

    day1_survey = df.iloc[:, 41:58].to_numpy(dtype=np.float32)
    day2_survey = df.iloc[:, 59:76].to_numpy(dtype=np.float32)
    day3_survey = df.iloc[:, 77:94].to_numpy(dtype=np.float32)

    day1_tp = df.iloc[:, 58].to_numpy(dtype=np.float32).reshape(-1, 1)
    day2_tp = df.iloc[:, 76].to_numpy(dtype=np.float32).reshape(-1, 1)

    day1_survey_z = token_scalers.survey_scaler.transform(day1_survey).astype(np.float32)
    day2_survey_z = token_scalers.survey_scaler.transform(day2_survey).astype(np.float32)
    day3_survey_z = token_scalers.survey_scaler.transform(day3_survey).astype(np.float32)

    ridge_z = token_scalers.ridge_scaler.transform(ridge_pred.reshape(-1, 1).astype(np.float32)).astype(np.float32)
    day1_tp_z = token_scalers.tp_scaler.transform(day1_tp).astype(np.float32)
    day2_tp_z = token_scalers.tp_scaler.transform(day2_tp).astype(np.float32)

    day1_token = np.hstack(
        [
            day1_survey_z,
            day1_tp_z,
            np.ones((len(df), 1), dtype=np.float32),
            np.zeros((len(df), 1), dtype=np.float32),
        ]
    )
    day2_token = np.hstack(
        [
            day2_survey_z,
            day2_tp_z,
            np.ones((len(df), 1), dtype=np.float32),
            np.full((len(df), 1), 0.5, dtype=np.float32),
        ]
    )
    day3_token = np.hstack(
        [
            day3_survey_z,
            np.zeros((len(df), 1), dtype=np.float32),
            np.zeros((len(df), 1), dtype=np.float32),
            np.ones((len(df), 1), dtype=np.float32),
        ]
    )

    ridge_raw = ridge_pred.reshape(-1, 1).astype(np.float32)
    scale_raw = (1.0 + day2_tp).astype(np.float32)
    y_raw = df.iloc[:, 94].to_numpy(dtype=np.float32).reshape(-1, 1) if df.shape[1] > 94 else None

    tensors = (
        torch.tensor(day1_token, dtype=torch.float32),
        torch.tensor(day2_token, dtype=torch.float32),
        torch.tensor(day3_token, dtype=torch.float32),
        torch.tensor(state, dtype=torch.float32),
        torch.tensor(ridge_z, dtype=torch.float32),
        torch.tensor(day1_tp_z, dtype=torch.float32),
        torch.tensor(day2_tp_z, dtype=torch.float32),
        torch.tensor(ridge_raw, dtype=torch.float32),
        torch.tensor(scale_raw, dtype=torch.float32),
    )
    if y_raw is not None:
        tensors = tensors + (torch.tensor(y_raw, dtype=torch.float32),)
    return tensors


class SharedDayEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualCorrector(nn.Module):
    """Structured residual head on top of Ridge.

    The model sees the day blocks explicitly, shares weights across days, and
    uses the raw 40-state one-hot block as an identity signal.
    """

    def __init__(self, token_dim: int = 20, hidden_dim: int = 96, head_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.day_encoder = SharedDayEncoder(token_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 6 + STATE_COUNT + 3, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim),
            nn.LayerNorm(head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        day1: torch.Tensor,
        day2: torch.Tensor,
        day3: torch.Tensor,
        state: torch.Tensor,
        ridge_z: torch.Tensor,
        day1_tp_z: torch.Tensor,
        day2_tp_z: torch.Tensor,
    ) -> torch.Tensor:
        e1 = self.day_encoder(day1)
        e2 = self.day_encoder(day2)
        e3 = self.day_encoder(day3)
        temporal = torch.cat(
            [
                e1,
                e2,
                e3,
                e2 - e1,
                e3 - e2,
                e3 - 2.0 * e2 + e1,
                state,
                ridge_z,
                day1_tp_z,
                day2_tp_z,
            ],
            dim=1,
        )
        return self.head(temporal)


def make_loader(*tensors: torch.Tensor, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
    )


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    targets = []
    for batch in loader:
        *inputs, ridge_raw, scale_raw, y_raw = batch
        inputs = [x.to(device) for x in inputs]
        ridge_raw = ridge_raw.to(device)
        scale_raw = scale_raw.to(device)
        y_raw = y_raw.to(device)
        pred_norm = model(*inputs)
        pred_raw = ridge_raw + pred_norm * scale_raw
        preds.append(pred_raw.detach().cpu().numpy())
        targets.append(y_raw.detach().cpu().numpy())
    return np.vstack(preds), np.vstack(targets)


def train_residual_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 400,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 40,
) -> Tuple[nn.Module, int, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=10,
        min_lr=1e-5,
    )

    best_state = None
    best_epoch = 0
    best_val = float("inf")
    bad_count = 0

    model.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            *inputs, ridge_raw, scale_raw, y_raw = batch
            inputs = [x.to(device) for x in inputs]
            ridge_raw = ridge_raw.to(device)
            scale_raw = scale_raw.to(device)
            y_raw = y_raw.to(device)

            pred_norm = model(*inputs)
            pred_raw = ridge_raw + pred_norm * scale_raw
            loss = F.mse_loss(pred_raw, y_raw)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_pred, val_y = predict_model(model, val_loader, device)
        val_mse = mean_squared_error(val_y.ravel(), val_pred.ravel())
        scheduler.step(val_mse)

        if val_mse < best_val - 1e-8:
            best_val = val_mse
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_count = 0
        else:
            bad_count += 1
            if bad_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_val


def tune_ridge_alpha(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[float, Ridge]:
    best_alpha = RIDGE_ALPHA_GRID[0]
    best_score = float("inf")
    best_model = None
    for alpha in RIDGE_ALPHA_GRID:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = mean_squared_error(y_val, pred)
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_model = model
    assert best_model is not None
    return best_alpha, best_model


def choose_blend_weight(y_true: np.ndarray, ridge_pred: np.ndarray, stack_pred: np.ndarray) -> float:
    best_w = 1.0
    best_score = float("inf")
    for w in BLEND_GRID:
        pred = w * stack_pred + (1.0 - w) * ridge_pred
        score = mean_squared_error(y_true, pred)
        if score < best_score:
            best_score = score
            best_w = float(w)
    return best_w


def main() -> None:
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    outer_train_df, outer_val_df = split_state_timewise(train_df, val_frac=0.2)

    y_outer_train = outer_train_df.iloc[:, 94].to_numpy(dtype=np.float32)
    y_outer_val = outer_val_df.iloc[:, 94].to_numpy(dtype=np.float32)

    X_outer_train, linear_scaler = build_linear_view(outer_train_df)
    X_outer_val, _ = build_linear_view(outer_val_df, linear_scaler)

    alpha, ridge_outer = tune_ridge_alpha(X_outer_train, y_outer_train, X_outer_val, y_outer_val)

    state_outer_train = state_ids(outer_train_df)
    ridge_outer_oof = fit_ridge_oof(X_outer_train, y_outer_train, state_outer_train, alpha=alpha, n_splits=5)
    ridge_outer_val = ridge_outer.predict(X_outer_val).astype(np.float32)

    token_scalers = fit_token_scalers(outer_train_df, ridge_outer_oof)
    outer_train_tensors = build_token_tensors(outer_train_df, ridge_outer_oof, token_scalers)
    outer_val_tensors = build_token_tensors(outer_val_df, ridge_outer_val, token_scalers)

    train_loader = make_loader(
        *outer_train_tensors[:-1],
        outer_train_tensors[-1],
        batch_size=64,
        shuffle=True,
        seed=SEED,
    )
    val_loader = make_loader(
        *outer_val_tensors[:-1],
        outer_val_tensors[-1],
        batch_size=256,
        shuffle=False,
        seed=SEED,
    )

    model = ResidualCorrector(token_dim=20, hidden_dim=96, head_dim=128, dropout=0.15)
    model, best_epoch, best_val = train_residual_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=500,
        lr=1e-3,
        weight_decay=1e-2,
        patience=50,
    )

    stack_outer_val_pred, outer_val_y = predict_model(model, val_loader, device)
    stack_outer_val_pred = stack_outer_val_pred.ravel()
    outer_val_y = outer_val_y.ravel()
    ridge_outer_val_raw = ridge_outer_val
    stack_outer_val_raw = stack_outer_val_pred
    blend = choose_blend_weight(outer_val_y, ridge_outer_val_raw, stack_outer_val_raw)

    print(f"Outer Ridge alpha: {alpha}")
    print(f"Outer stacked val MSE: {mean_squared_error(outer_val_y, stack_outer_val_raw):.5f}")
    print(f"Outer ridge val MSE: {mean_squared_error(outer_val_y, ridge_outer_val_raw):.5f}")
    print(f"Chosen blend weight: {blend:.2f}")
    print(f"Best epoch from outer training: {best_epoch}")

    # Refit on the full training data for the final submission.
    full_y = train_df.iloc[:, 94].to_numpy(dtype=np.float32)
    full_X, full_linear_scaler = build_linear_view(train_df)
    full_state = state_ids(train_df)
    ridge_full_oof = fit_ridge_oof(full_X, full_y, full_state, alpha=alpha, n_splits=5)
    ridge_full = Ridge(alpha=alpha)
    ridge_full.fit(full_X, full_y)

    full_token_scalers = fit_token_scalers(train_df, ridge_full_oof)
    full_train_tensors = build_token_tensors(train_df, ridge_full_oof, full_token_scalers)
    full_train_loader = make_loader(
        *full_train_tensors[:-1],
        full_train_tensors[-1],
        batch_size=64,
        shuffle=True,
        seed=SEED,
    )

    final_model = ResidualCorrector(token_dim=20, hidden_dim=96, head_dim=128, dropout=0.15)
    final_model = final_model.to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1e-3, weight_decay=1e-2)

    final_model.train()
    for _ in range(max(best_epoch, 100)):
        for batch in full_train_loader:
            *inputs, ridge_raw, scale_raw, y_raw = batch
            inputs = [x.to(device) for x in inputs]
            ridge_raw = ridge_raw.to(device)
            scale_raw = scale_raw.to(device)
            y_raw = y_raw.to(device)

            pred_norm = final_model(*inputs)
            pred_raw = ridge_raw + pred_norm * scale_raw
            loss = F.mse_loss(pred_raw, y_raw)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # Build test features.
    X_test, _ = build_linear_view(test_df, full_linear_scaler)
    ridge_test = ridge_full.predict(X_test).astype(np.float32)
    test_tensors = build_token_tensors(test_df, ridge_test, full_token_scalers)
    test_loader = make_loader(
        *test_tensors,
        torch.zeros((len(test_df), 1), dtype=torch.float32),
        batch_size=256,
        shuffle=False,
        seed=SEED,
    )

    final_model.eval()
    test_preds = []
    with torch.no_grad():
        for batch in test_loader:
            *inputs, ridge_raw, scale_raw, _dummy = batch
            inputs = [x.to(device) for x in inputs]
            ridge_raw = ridge_raw.to(device)
            scale_raw = scale_raw.to(device)
            pred_norm = final_model(*inputs)
            pred_raw = ridge_raw + pred_norm * scale_raw
            test_preds.append(pred_raw.detach().cpu().numpy())

    stack_test = np.vstack(test_preds).ravel()
    final_test = blend * stack_test + (1.0 - blend) * ridge_test

    submission = pd.DataFrame(
        {
            "id": test_df.iloc[:, 0].values,
            "tested_positive": final_test,
        }
    )
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Saved {SUBMISSION_PATH.resolve()}")


if __name__ == "__main__":
    main()
