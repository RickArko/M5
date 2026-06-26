"""Forecast models for the M5 task."""

from m5.models.hierarchical import (
    build_hier_base_forecaster,
    build_hier_reconcilers,
    fit_predict_hier,
)
from m5.models.lgbm import build_lgbm_forecaster, fit_lgbm, fit_predict_lgbm
from m5.models.segmented import (
    fit_predict_segmented,
    fit_predict_store,
    fit_predict_store_cat,
    fit_predict_store_dept,
    segmented_cv,
    store_cat_cv,
    store_cv,
    store_dept_cv,
)
from m5.models.stats import build_stats_forecaster, fit_predict_stats
from m5.models.toto import toto_cv, toto_forecast

try:
    from m5.models.bayesian import (
        bayesian_cv,
        bayesian_routed_cv,
        classify_intermittency,
        encode_hier_panel,
        fit_bayes_hier_zinb,
        fit_predict_bayes,
        fit_predict_bayes_negbin,
        fit_predict_bayes_zinb,
        intermittency_profiles,
        pick_intermittency_examples,
        posterior_mean_forecast_hier_zinb,
        posterior_mean_forecast_negbin,
        posterior_mean_forecast_zinb,
        posterior_zero_prob_zinb,
        route_demand_class,
        series_zero_rate,
    )
except ImportError:
    bayesian_cv = None
    bayesian_routed_cv = None
    classify_intermittency = None
    encode_hier_panel = None
    fit_bayes_hier_zinb = None
    fit_predict_bayes = None
    fit_predict_bayes_negbin = None
    fit_predict_bayes_zinb = None
    intermittency_profiles = None
    pick_intermittency_examples = None
    posterior_mean_forecast_hier_zinb = None
    posterior_mean_forecast_negbin = None
    posterior_mean_forecast_zinb = None
    posterior_zero_prob_zinb = None
    route_demand_class = None
    series_zero_rate = None

__all__ = [
    "bayesian_cv",
    "bayesian_routed_cv",
    "build_hier_base_forecaster",
    "build_hier_reconcilers",
    "build_lgbm_forecaster",
    "build_stats_forecaster",
    "classify_intermittency",
    "encode_hier_panel",
    "fit_bayes_hier_zinb",
    "fit_lgbm",
    "fit_predict_bayes",
    "fit_predict_bayes_negbin",
    "fit_predict_bayes_zinb",
    "fit_predict_hier",
    "fit_predict_lgbm",
    "fit_predict_segmented",
    "fit_predict_stats",
    "fit_predict_store",
    "fit_predict_store_cat",
    "fit_predict_store_dept",
    "intermittency_profiles",
    "pick_intermittency_examples",
    "posterior_mean_forecast_hier_zinb",
    "posterior_mean_forecast_negbin",
    "posterior_mean_forecast_zinb",
    "posterior_zero_prob_zinb",
    "route_demand_class",
    "segmented_cv",
    "series_zero_rate",
    "store_cat_cv",
    "store_cv",
    "store_dept_cv",
    "toto_cv",
    "toto_forecast",
]
