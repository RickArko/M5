"""Bayesian count GLMs for capped M5 experiments (NegBin + zero-inflated NegBin).

Requires the optional ``bayesian`` dependency group::

    uv sync --group bayesian

MCMC does not scale to 30k series — use ``n_series`` caps and treat outputs as
research / interval forecasts, not a production replacement for LightGBM.
"""

from __future__ import annotations

import logging
import os
import warnings
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from m5.config import REPO_ROOT, SETTINGS, set_global_seed
from m5.features import add_date_features, add_event_flag, add_price_features, add_snap_flag
from m5.logging import logger

if TYPE_CHECKING:
    import arviz as az

BayesLikelihood = Literal["negbin", "zinb"]
IntermittencyClass = Literal["smooth", "intermittent", "erratic", "lumpy"]
_FORECAST_COL = {"negbin": "Bayes-NegBin", "zinb": "Bayes-ZINB", "routed": "Bayes-Routed"}

# Syntetos-Boylan ADI / CV2 thresholds (standard intermittent-demand taxonomy).
ADI_THRESH: float = 1.32
CV2_THRESH: float = 0.49
_MIN_HIER_SERIES = 2

_MIN_TRAIN_DAYS = 56
_FEATURE_FNS = (add_date_features, add_snap_flag, add_event_flag, add_price_features)


def _require_pymc() -> tuple[Any, Any]:
    try:
        import arviz as az
        import pymc as pm
    except ImportError as exc:
        msg = "Bayesian models require optional deps — run: uv sync --group bayesian"
        raise ImportError(msg) from exc
    _ensure_pytensor_cache()
    return pm, az


def _ensure_pytensor_cache() -> None:
    cache = REPO_ROOT / ".pytensor"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("PYTENSOR_FLAGS", f"compiledir={cache}")


@contextmanager
def quiet_mcmc():
    """Silence PyMC sampling chatter — intended for batched CV, not exploratory notebooks."""
    loggers = [logging.getLogger(name) for name in ("pymc", "pytensor", "arviz")]
    prev_levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.ERROR)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        yield
    for lg, level in zip(loggers, prev_levels, strict=True):
        lg.setLevel(level)


def add_minimal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the repo's lightweight calendar / price features."""
    out = df.sort_values(["unique_id", "ds"]).copy()
    for fn in _FEATURE_FNS:
        out = fn(out)
    return out


def series_zero_rate(df: pd.DataFrame, *, id_col: str = "unique_id", target_col: str = "y") -> pd.Series:
    """Fraction of zero sales days per series."""
    return df.groupby(id_col, observed=True)[target_col].apply(lambda s: float((s == 0).mean()))


def pick_intermittency_examples(
    df: pd.DataFrame,
    *,
    min_train_days: int = _MIN_TRAIN_DAYS,
) -> tuple[str, str]:
    """Return (high_zero_rate_id, low_zero_rate_id) with enough history for MCMC."""
    counts = df.groupby("unique_id", observed=True).agg(
        n_days=("ds", "size"), zero_rate=("y", lambda s: (s == 0).mean())
    )
    eligible = counts.loc[counts["n_days"] >= min_train_days + SETTINGS.horizon]
    if len(eligible) < 2:
        msg = "Need at least two series with sufficient history for intermittency comparison."
        raise ValueError(msg)
    high = eligible["zero_rate"].idxmax()
    low = eligible["zero_rate"].idxmin()
    return str(high), str(low)


def _series_adi(y: np.ndarray) -> float:
    """Average demand interval — mean gap in days between non-zero sales."""
    nonzero = np.flatnonzero(y > 0)
    if len(nonzero) < 2:
        return float("inf")
    return float(np.diff(nonzero).mean())


def _series_cv2(y: np.ndarray) -> float:
    """Squared coefficient of variation on non-zero demand."""
    nz = y[y > 0]
    if len(nz) < 2:
        return float("inf")
    mean = float(nz.mean())
    if mean == 0.0:
        return float("inf")
    return float((nz.std(ddof=1) / mean) ** 2)


def classify_intermittency(
    adi: float,
    cv2: float,
    *,
    adi_thresh: float = ADI_THRESH,
    cv2_thresh: float = CV2_THRESH,
) -> IntermittencyClass:
    """Map ADI / CV² to smooth, intermittent, erratic, or lumpy (Syntetos–Boylan)."""
    high_adi = adi >= adi_thresh
    high_cv2 = cv2 >= cv2_thresh
    if not high_adi and not high_cv2:
        return "smooth"
    if high_adi and not high_cv2:
        return "intermittent"
    if not high_adi and high_cv2:
        return "erratic"
    return "lumpy"


def intermittency_profiles(
    df: pd.DataFrame,
    *,
    id_col: str = "unique_id",
    target_col: str = "y",
    adi_thresh: float = ADI_THRESH,
    cv2_thresh: float = CV2_THRESH,
) -> pd.DataFrame:
    """Per-series ADI, CV², zero-rate, and demand class."""
    rows: list[dict[str, object]] = []
    for uid, grp in df.groupby(id_col, observed=True):
        y = grp[target_col].to_numpy(dtype=float)
        adi = _series_adi(y)
        cv2 = _series_cv2(y)
        rows.append(
            {
                id_col: uid,
                "adi": adi,
                "cv2": cv2,
                "zero_rate": float((y == 0).mean()),
                "demand_class": classify_intermittency(
                    adi, cv2, adi_thresh=adi_thresh, cv2_thresh=cv2_thresh
                ),
            }
        )
    out = pd.DataFrame(rows).set_index(id_col)
    return out.sort_values("zero_rate", ascending=False)


def route_demand_class(demand_class: IntermittencyClass) -> tuple[BayesLikelihood, bool]:
    """Return (likelihood, use_hierarchical) for a demand class.

    Smooth / erratic → univariate NegBin.  Intermittent / lumpy → hierarchical ZINB
    when enough peers exist in the same pool, else univariate ZINB.
    """
    if demand_class in ("smooth", "erratic"):
        return "negbin", False
    return "zinb", True


@dataclass(frozen=True)
class HierPanelEncoding:
    """Index maps for a multi-series hierarchical panel."""

    df: pd.DataFrame
    series_map: dict[str, int]
    pool_map: dict[str, int]
    series_pool: dict[str, int]

    @property
    def n_series(self) -> int:
        return len(self.series_map)

    @property
    def n_pools(self) -> int:
        return len(self.pool_map)


def encode_hier_panel(df: pd.DataFrame, *, pool_col: str = "dept_id") -> HierPanelEncoding:
    """Add ``series_idx`` / ``pool_idx`` columns for hierarchical PyMC models."""
    if pool_col not in df.columns:
        msg = f"Pool column {pool_col!r} missing — need static column on long frame."
        raise KeyError(msg)

    out = add_minimal_features(df)
    series_ids = sorted(out["unique_id"].astype(str).unique())
    series_map = {uid: i for i, uid in enumerate(series_ids)}
    out["series_idx"] = out["unique_id"].astype(str).map(series_map).astype(int)

    static = out.groupby("unique_id", observed=True)[pool_col].first().astype(str)
    pool_levels = sorted(static.unique())
    pool_map = {p: i for i, p in enumerate(pool_levels)}
    series_pool = {uid: pool_map[static.loc[uid]] for uid in series_ids}
    out["pool_idx"] = out["unique_id"].astype(str).map(series_pool).astype(int)
    return HierPanelEncoding(out, series_map, pool_map, series_pool)


def posterior_mean_forecast_hier_zinb(
    post: dict[str, np.ndarray],
    future: pd.DataFrame,
    *,
    series_idx: int,
) -> np.ndarray:
    """Posterior mean forecast for one series from a hierarchical ZINB fit."""
    dow, price, snap, event = _future_covariates(future)
    logit_psi = (
        post["gamma0"][:, series_idx][:, None]
        + post["gamma_dow"][:, dow]
        + post["gamma_snap"][:, None] * snap[None, :]
        + post["gamma_event"][:, None] * event[None, :]
    )
    psi = 1.0 / (1.0 + np.exp(-logit_psi))
    log_mu = (
        post["beta0"][:, series_idx][:, None]
        + post["beta_dow"][:, dow]
        + post["beta_price"][:, None] * price[None, :]
    )
    mu = np.exp(log_mu)
    return ((1.0 - psi) * mu).mean(axis=0)


def fit_bayes_hier_zinb(
    panel: pd.DataFrame | HierPanelEncoding,
    *,
    pool_col: str = "dept_id",
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    seed: int | None = None,
    progressbar: bool = False,
    quiet: bool = False,
    compute_convergence_checks: bool | None = None,
) -> tuple[az.InferenceData, HierPanelEncoding]:
    """Hierarchical ZINB with partial pooling on ``gamma0`` and ``beta0`` by *pool_col*."""
    pm, _ = _require_pymc()
    seed = SETTINGS.seed if seed is None else seed
    if compute_convergence_checks is None:
        compute_convergence_checks = not quiet

    enc = panel if isinstance(panel, HierPanelEncoding) else encode_hier_panel(panel, pool_col=pool_col)
    train = enc.df
    n_series = enc.n_series
    n_pools = enc.n_pools
    pool_by_series = np.array(
        [enc.series_pool[uid] for uid, _ in sorted(enc.series_map.items(), key=lambda kv: kv[1])]
    )

    coords = {"dayofweek": np.arange(7), "series": np.arange(n_series), "pool": np.arange(n_pools)}
    with pm.Model(coords=coords):
        series_idx = pm.Data("series_idx", train["series_idx"].to_numpy())
        dow = pm.Data("dow", train["dayofweek"].to_numpy())
        price = pm.Data("price", train["price_norm"].to_numpy())
        snap = pm.Data("snap", train["snap"].to_numpy())
        event = pm.Data("event", train["is_event"].to_numpy())

        mu_gamma0 = pm.Normal("mu_gamma0", 0, 1, dims="pool")
        sigma_gamma0 = pm.HalfNormal("sigma_gamma0", 0.5)
        gamma0 = pm.Normal("gamma0", mu_gamma0[pool_by_series], sigma_gamma0, dims="series")

        mu_beta0 = pm.Normal("mu_beta0", 0, 1, dims="pool")
        sigma_beta0 = pm.HalfNormal("sigma_beta0", 0.5)
        beta0 = pm.Normal("beta0", mu_beta0[pool_by_series], sigma_beta0, dims="series")

        gamma_dow = pm.Normal("gamma_dow", 0, 0.5, dims="dayofweek")
        gamma_snap = pm.Normal("gamma_snap", 0, 0.5)
        gamma_event = pm.Normal("gamma_event", 0, 0.5)

        beta_dow = pm.Normal("beta_dow", 0, 0.5, dims="dayofweek")
        beta_price = pm.Normal("beta_price", 0, 0.5)
        alpha = pm.Exponential("alpha", 1.0)

        logit_psi = gamma0[series_idx] + gamma_dow[dow] + gamma_snap * snap + gamma_event * event
        psi = pm.Deterministic("psi", pm.math.sigmoid(logit_psi))
        log_mu = beta0[series_idx] + beta_dow[dow] + beta_price * price
        mu = pm.Deterministic("mu", pm.math.exp(log_mu))
        pm.ZeroInflatedNegativeBinomial("y", psi=psi, mu=mu, alpha=alpha, observed=train["y"].to_numpy())

        idata = _sample_idata(
            draws=draws,
            tune=tune,
            chains=chains,
            seed=seed,
            progressbar=progressbar,
            compute_convergence_checks=compute_convergence_checks,
            quiet=quiet,
        )
    return idata, enc


def _as_draw_matrix(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
        return arr.T
    return arr


def extract_posterior(
    idata: az.InferenceData, *, var_names: list[str] | None = None
) -> dict[str, np.ndarray]:
    """Stack MCMC chains into flat draw arrays for vectorised forecast math."""
    _, az = _require_pymc()
    post = az.extract(idata, var_names=var_names)
    out: dict[str, np.ndarray] = {}

    if not hasattr(post, "data_vars"):
        name = post.name or (var_names[0] if var_names else "unknown")
        return {name: _as_draw_matrix(post.values)}

    for name in post.data_vars:
        out[str(name)] = _as_draw_matrix(post[name].values)
    return out


def _future_covariates(future: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if "dayofweek" not in future.columns:
        future = add_minimal_features(future)
    return (
        future["dayofweek"].to_numpy(),
        future["price_norm"].to_numpy(),
        future["snap"].to_numpy(),
        future["is_event"].to_numpy(),
    )


def posterior_mean_forecast_negbin(post: dict[str, np.ndarray], future: pd.DataFrame) -> np.ndarray:
    """Point forecast = mean over draws of ``exp(log_mu)``."""
    dow, price, snap, _event = _future_covariates(future)
    beta_dow = post["beta_dow"]
    log_mu = (
        post["beta0"][:, None]
        + beta_dow[:, dow]
        + post["beta_price"][:, None] * price[None, :]
        + post["beta_snap"][:, None] * snap[None, :]
    )
    return np.exp(log_mu).mean(axis=0)


def posterior_mean_forecast_zinb(post: dict[str, np.ndarray], future: pd.DataFrame) -> np.ndarray:
    """Point forecast = mean over draws of ``(1 - psi) * mu`` (PyMC ZINB mean)."""
    dow, price, snap, event = _future_covariates(future)
    logit_psi = (
        post["gamma0"][:, None]
        + post["gamma_dow"][:, dow]
        + post["gamma_snap"][:, None] * snap[None, :]
        + post["gamma_event"][:, None] * event[None, :]
    )
    psi = 1.0 / (1.0 + np.exp(-logit_psi))
    log_mu = post["beta0"][:, None] + post["beta_dow"][:, dow] + post["beta_price"][:, None] * price[None, :]
    mu = np.exp(log_mu)
    return ((1.0 - psi) * mu).mean(axis=0)


def posterior_zero_prob_zinb(post: dict[str, np.ndarray], future: pd.DataFrame) -> np.ndarray:
    """Posterior mean of P(y=0) for ZINB: ``psi + (1-psi)*NB(0|mu,alpha)``."""
    dow, price, snap, event = _future_covariates(future)
    logit_psi = (
        post["gamma0"][:, None]
        + post["gamma_dow"][:, dow]
        + post["gamma_snap"][:, None] * snap[None, :]
        + post["gamma_event"][:, None] * event[None, :]
    )
    psi = 1.0 / (1.0 + np.exp(-logit_psi))
    log_mu = post["beta0"][:, None] + post["beta_dow"][:, dow] + post["beta_price"][:, None] * price[None, :]
    mu = np.exp(log_mu)
    alpha = post["alpha"][:, None]
    nb_zero = np.power(alpha / (alpha + mu), alpha)
    p_zero = psi + (1.0 - psi) * nb_zero
    return p_zero.mean(axis=0)


def _sample_idata(
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    progressbar: bool,
    compute_convergence_checks: bool,
    quiet: bool,
) -> az.InferenceData:
    pm, _ = _require_pymc()
    sample_ctx = quiet_mcmc() if quiet else nullcontext()
    with sample_ctx:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            target_accept=0.9,
            random_seed=seed,
            progressbar=progressbar,
            compute_convergence_checks=compute_convergence_checks,
        )


def fit_bayes_zinb(
    train: pd.DataFrame,
    *,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    seed: int | None = None,
    progressbar: bool = False,
    quiet: bool = False,
    compute_convergence_checks: bool | None = None,
) -> az.InferenceData:
    """Fit ZINB and return the full ``InferenceData`` (for trace / calibration plots)."""
    pm, _ = _require_pymc()
    seed = SETTINGS.seed if seed is None else seed
    if compute_convergence_checks is None:
        compute_convergence_checks = not quiet

    train = add_minimal_features(train)
    coords = {"dayofweek": np.arange(7)}
    with pm.Model(coords=coords):
        dow = pm.Data("dow", train["dayofweek"].to_numpy())
        price = pm.Data("price", train["price_norm"].to_numpy())
        snap = pm.Data("snap", train["snap"].to_numpy())
        event = pm.Data("event", train["is_event"].to_numpy())

        gamma0 = pm.Normal("gamma0", 0, 1)
        gamma_dow = pm.Normal("gamma_dow", 0, 0.5, dims="dayofweek")
        gamma_snap = pm.Normal("gamma_snap", 0, 0.5)
        gamma_event = pm.Normal("gamma_event", 0, 0.5)

        beta0 = pm.Normal("beta0", 0, 1)
        beta_dow = pm.Normal("beta_dow", 0, 0.5, dims="dayofweek")
        beta_price = pm.Normal("beta_price", 0, 0.5)
        alpha = pm.Exponential("alpha", 1.0)

        logit_psi = gamma0 + gamma_dow[dow] + gamma_snap * snap + gamma_event * event
        psi = pm.Deterministic("psi", pm.math.sigmoid(logit_psi))
        log_mu = beta0 + beta_dow[dow] + beta_price * price
        mu = pm.Deterministic("mu", pm.math.exp(log_mu))
        pm.ZeroInflatedNegativeBinomial("y", psi=psi, mu=mu, alpha=alpha, observed=train["y"].to_numpy())

        return _sample_idata(
            draws=draws,
            tune=tune,
            chains=chains,
            seed=seed,
            progressbar=progressbar,
            compute_convergence_checks=compute_convergence_checks,
            quiet=quiet,
        )


def fit_bayes_negbin(
    train: pd.DataFrame,
    *,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    seed: int | None = None,
    progressbar: bool = False,
    quiet: bool = False,
    compute_convergence_checks: bool | None = None,
) -> az.InferenceData:
    """Fit NegBin and return the full ``InferenceData`` (for trace / calibration plots)."""
    pm, _ = _require_pymc()
    seed = SETTINGS.seed if seed is None else seed
    if compute_convergence_checks is None:
        compute_convergence_checks = not quiet

    train = add_minimal_features(train)
    coords = {"dayofweek": np.arange(7)}
    with pm.Model(coords=coords):
        dow = pm.Data("dow", train["dayofweek"].to_numpy())
        price = pm.Data("price", train["price_norm"].to_numpy())
        snap = pm.Data("snap", train["snap"].to_numpy())

        beta0 = pm.Normal("beta0", 0, 1)
        beta_dow = pm.Normal("beta_dow", 0, 0.5, dims="dayofweek")
        beta_price = pm.Normal("beta_price", 0, 0.5)
        beta_snap = pm.Normal("beta_snap", 0, 0.5)
        alpha = pm.Exponential("alpha", 1.0)

        log_mu = beta0 + beta_dow[dow] + beta_price * price + beta_snap * snap
        mu = pm.Deterministic("mu", pm.math.exp(log_mu))
        pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=train["y"].to_numpy())

        return _sample_idata(
            draws=draws,
            tune=tune,
            chains=chains,
            seed=seed,
            progressbar=progressbar,
            compute_convergence_checks=compute_convergence_checks,
            quiet=quiet,
        )


def fit_predict_bayes_negbin(
    train: pd.DataFrame,
    future: pd.DataFrame,
    *,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    seed: int | None = None,
    progressbar: bool = False,
    quiet: bool = False,
    compute_convergence_checks: bool | None = None,
) -> np.ndarray:
    """Fit a univariate NegBin GLM and return posterior-mean forecasts for *future*."""
    seed = SETTINGS.seed if seed is None else seed
    if compute_convergence_checks is None:
        compute_convergence_checks = not quiet

    train = add_minimal_features(train)
    future = add_minimal_features(future)
    idata = fit_bayes_negbin(
        train,
        draws=draws,
        tune=tune,
        chains=chains,
        seed=seed,
        progressbar=progressbar,
        quiet=quiet,
        compute_convergence_checks=compute_convergence_checks,
    )
    return posterior_mean_forecast_negbin(extract_posterior(idata), future)


def fit_predict_bayes_zinb(
    train: pd.DataFrame,
    future: pd.DataFrame,
    *,
    draws: int = 400,
    tune: int = 400,
    chains: int = 2,
    seed: int | None = None,
    progressbar: bool = False,
    quiet: bool = False,
    compute_convergence_checks: bool | None = None,
) -> np.ndarray:
    """Fit a zero-inflated NegBin GLM (occurrence + size) and forecast *future*.

    Occurrence (logit ``psi``): day-of-week, SNAP, event flag.
    Size (log ``mu``): day-of-week, normalised price.
    """
    seed = SETTINGS.seed if seed is None else seed
    if compute_convergence_checks is None:
        compute_convergence_checks = not quiet

    train = add_minimal_features(train)
    future = add_minimal_features(future)
    idata = fit_bayes_zinb(
        train,
        draws=draws,
        tune=tune,
        chains=chains,
        seed=seed,
        progressbar=progressbar,
        quiet=quiet,
        compute_convergence_checks=compute_convergence_checks,
    )
    return posterior_mean_forecast_zinb(extract_posterior(idata), future)


def fit_predict_bayes(
    train: pd.DataFrame,
    future: pd.DataFrame,
    *,
    likelihood: BayesLikelihood = "negbin",
    **kwargs: Any,
) -> np.ndarray:
    """Dispatch to NegBin or zero-inflated NegBin fit/predict."""
    if likelihood == "negbin":
        return fit_predict_bayes_negbin(train, future, **kwargs)
    if likelihood == "zinb":
        return fit_predict_bayes_zinb(train, future, **kwargs)
    raise ValueError(f"Unknown likelihood {likelihood!r}; use 'negbin' or 'zinb'.")


def _select_series_ids(df: pd.DataFrame, n_series: int, series_ids: list[str] | None) -> list[str]:
    if series_ids is not None:
        return list(series_ids)[:n_series]
    vol = df.groupby("unique_id", observed=True)["y"].sum()
    return vol.nlargest(n_series).index.astype(str).tolist()


def bayesian_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = 1,
    step_size: int | None = None,
    n_series: int = 20,
    series_ids: list[str] | None = None,
    draws: int = 300,
    tune: int = 300,
    chains: int = 2,
    min_train_days: int = _MIN_TRAIN_DAYS,
    quiet: bool = True,
    likelihood: BayesLikelihood = "negbin",
) -> pd.DataFrame:
    """Rolling-origin CV with a Bayesian count GLM per series.

    Returns a Nixtla CV frame with ``Bayes-NegBin`` or ``Bayes-ZINB`` so
    :func:`m5.evaluation.wrmsse_for_models` can score it beside stats / LGBM.

    Set ``likelihood='zinb'`` for zero-inflated series (SNAP/event on occurrence).
    """
    set_global_seed()
    step = step_size or h
    featured = add_minimal_features(df)
    ids = _select_series_ids(featured, n_series, series_ids)
    panel = featured.loc[featured["unique_id"].astype(str).isin(ids)].copy()
    max_ds = panel["ds"].max()
    forecast_col = _FORECAST_COL[likelihood]

    logger.info(
        f"bayesian_cv: likelihood={likelihood} h={h} n_windows={n_windows} step={step} "
        f"series={len(ids)} draws={draws} tune={tune}"
    )

    records: list[dict[str, object]] = []
    for w in range(n_windows):
        cutoff = max_ds - pd.Timedelta(days=(n_windows - 1 - w) * step + h)
        forecast_start = cutoff + pd.Timedelta(days=1)
        forecast_dates = pd.date_range(forecast_start, periods=h, freq="D")

        for i, uid in enumerate(ids, start=1):
            sdf = panel.loc[panel["unique_id"].astype(str) == uid]
            train = sdf.loc[sdf["ds"] <= cutoff]
            future = sdf.loc[sdf["ds"].isin(forecast_dates)]
            if len(train) < min_train_days or len(future) != h:
                logger.warning(f"bayesian_cv: skipping {uid} — insufficient train/future rows")
                continue

            y_hat = fit_predict_bayes(
                train,
                future,
                likelihood=likelihood,
                draws=draws,
                tune=tune,
                chains=chains,
                progressbar=False,
                quiet=quiet,
            )
            truth = future.set_index("ds")["y"]
            for k, day in enumerate(forecast_dates):
                records.append(
                    {
                        "unique_id": uid,
                        "ds": day,
                        "cutoff": cutoff,
                        "y": float(truth.loc[day]),
                        forecast_col: float(y_hat[k]),
                    }
                )

            if i % max(1, len(ids) // 5) == 0 or i == len(ids):
                logger.info(f"bayesian_cv window {w + 1}/{n_windows}: {i}/{len(ids)} series done")

    if not records:
        raise ValueError("bayesian_cv produced no forecasts — check n_series and history length.")

    cv_df = pd.DataFrame(records)
    if "unique_id" in df.columns:
        cv_df["unique_id"] = cv_df["unique_id"].astype(df["unique_id"].dtype)
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])
    return cv_df


def bayesian_routed_cv(
    df: pd.DataFrame,
    *,
    h: int = SETTINGS.horizon,
    n_windows: int = 1,
    step_size: int | None = None,
    n_series: int = 20,
    series_ids: list[str] | None = None,
    pool_col: str = "dept_id",
    min_hier_series: int = _MIN_HIER_SERIES,
    draws: int = 300,
    tune: int = 300,
    chains: int = 2,
    min_train_days: int = _MIN_TRAIN_DAYS,
    quiet: bool = True,
) -> pd.DataFrame:
    """ADI/CV²-routed CV: NegBin for smooth/erratic, hierarchical ZINB for intermittent/lumpy.

    Returns ``unique_id, ds, cutoff, y, Bayes-Routed, demand_class, route``.
    """
    set_global_seed()
    if pool_col not in df.columns:
        msg = f"pool_col {pool_col!r} not in dataframe — required for hierarchical pooling."
        raise KeyError(msg)

    step = step_size or h
    featured = add_minimal_features(df)
    ids = _select_series_ids(featured, n_series, series_ids)
    panel = featured.loc[featured["unique_id"].astype(str).isin(ids)].copy()
    max_ds = panel["ds"].max()

    logger.info(
        f"bayesian_routed_cv: h={h} n_windows={n_windows} pool={pool_col} series={len(ids)} draws={draws}"
    )

    records: list[dict[str, object]] = []
    for w in range(n_windows):
        cutoff = max_ds - pd.Timedelta(days=(n_windows - 1 - w) * step + h)
        forecast_start = cutoff + pd.Timedelta(days=1)
        forecast_dates = pd.date_range(forecast_start, periods=h, freq="D")
        train_all = panel.loc[panel["ds"] <= cutoff]
        profiles = intermittency_profiles(train_all)

        hier_cache: dict[tuple[str, str], tuple[Any, HierPanelEncoding]] = {}

        for i, uid in enumerate(ids, start=1):
            uid_s = str(uid)
            train_i = train_all.loc[train_all["unique_id"].astype(str) == uid_s]
            future_i = panel.loc[
                (panel["unique_id"].astype(str) == uid_s) & (panel["ds"].isin(forecast_dates))
            ]
            if len(train_i) < min_train_days or len(future_i) != h:
                logger.warning(f"bayesian_routed_cv: skipping {uid_s} — insufficient rows")
                continue

            demand_class = (
                profiles.loc[uid_s, "demand_class"]
                if uid_s in profiles.index
                else profiles.loc[uid, "demand_class"]
            )
            likelihood, use_hier = route_demand_class(demand_class)
            pool_val = str(train_i[pool_col].iloc[0])
            route_label = f"{likelihood}"

            if use_hier:
                cache_key = (pool_val, demand_class)
                if cache_key not in hier_cache:
                    peer_ids = [
                        str(u)
                        for u in profiles.index
                        if str(u) in ids
                        and profiles.loc[u, "demand_class"] == demand_class
                        and str(train_all.loc[train_all["unique_id"].astype(str) == str(u), pool_col].iloc[0])
                        == pool_val
                    ]
                    if len(peer_ids) >= min_hier_series:
                        hier_train = train_all.loc[train_all["unique_id"].astype(str).isin(peer_ids)]
                        idata, enc = fit_bayes_hier_zinb(
                            hier_train,
                            pool_col=pool_col,
                            draws=draws,
                            tune=tune,
                            chains=chains,
                            progressbar=False,
                            quiet=quiet,
                        )
                        hier_cache[cache_key] = (idata, enc)
                        route_label = f"hier-{likelihood}"

                if cache_key in hier_cache:
                    idata, enc = hier_cache[cache_key]
                    post = extract_posterior(idata)
                    sidx = enc.series_map[uid_s]
                    y_hat = posterior_mean_forecast_hier_zinb(post, future_i, series_idx=sidx)
                else:
                    y_hat = fit_predict_bayes_zinb(
                        train_i, future_i, draws=draws, tune=tune, chains=chains, quiet=quiet
                    )
                    route_label = f"uni-{likelihood}"
            elif likelihood == "negbin":
                y_hat = fit_predict_bayes_negbin(
                    train_i, future_i, draws=draws, tune=tune, chains=chains, quiet=quiet
                )
            else:
                y_hat = fit_predict_bayes_zinb(
                    train_i, future_i, draws=draws, tune=tune, chains=chains, quiet=quiet
                )

            truth = future_i.set_index("ds")["y"]
            for k, day in enumerate(forecast_dates):
                records.append(
                    {
                        "unique_id": uid_s,
                        "ds": day,
                        "cutoff": cutoff,
                        "y": float(truth.loc[day]),
                        "Bayes-Routed": float(y_hat[k]),
                        "demand_class": demand_class,
                        "route": route_label,
                    }
                )

            if i % max(1, len(ids) // 5) == 0 or i == len(ids):
                logger.info(f"bayesian_routed_cv window {w + 1}/{n_windows}: {i}/{len(ids)} series done")

    if not records:
        raise ValueError("bayesian_routed_cv produced no forecasts.")

    cv_df = pd.DataFrame(records)
    if "unique_id" in df.columns:
        cv_df["unique_id"] = cv_df["unique_id"].astype(df["unique_id"].dtype)
    cv_df["ds"] = pd.to_datetime(cv_df["ds"])
    cv_df["cutoff"] = pd.to_datetime(cv_df["cutoff"])
    return cv_df
