import json
from importlib.resources import as_file, files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMRegressor
from skforecast.direct import ForecasterDirect
from skforecast.model_selection import TimeSeriesFold, backtesting_forecaster
from skforecast.preprocessing import RollingFeatures
from skforecast.recursive import ForecasterEquivalentDate

import data
import presentation

# -----------------------------------------------------------------------------
# Data loading & preparation
# -----------------------------------------------------------------------------

data_source = files(data).joinpath("2025-08-25T09_06_06+00_00_3zyjl.csv")
presentation_output = files(presentation)

with as_file(data_source) as csv_path:
    power_by_minute = (
        pd.read_csv(csv_path, parse_dates=["timestamp_local"])
        .set_index("timestamp_local")
        .sort_index()
        .asfreq("min")
        .fillna(
            0.0
        )  # Missing timestamps are assumed to correspond to zero consumption.
        .rename(columns={"power_kw_avg": "power_kw"})
    )

# -----------------------------------------------------------------------------
# Exploratory analysis
# -----------------------------------------------------------------------------

power_by_day = power_by_minute["power_kw"].resample("D").mean()

POWER_ANALYSIS_PLOT = "power_analysis_plots.png"

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Power Consumption Analysis", fontsize=16)

sns.lineplot(ax=axes[0, 0], data=power_by_day.values)
axes[0, 0].set_title("Daily Average Power")
axes[0, 0].set_xlabel("Date")
axes[0, 0].set_ylabel("Power (kW)")

sns.histplot(ax=axes[0, 1], data=power_by_minute["power_kw"], bins=50, kde=True)
axes[0, 1].set_title("Distribution of Minute-Level Power")
axes[0, 1].set_xlabel("Power (kW)")
axes[0, 1].set_ylabel("Frequency")

tmp = power_by_minute.assign(
    weekday=lambda df: df.index.weekday,
    hour=lambda df: df.index.hour,
)

sns.boxplot(ax=axes[1, 0], x=tmp["weekday"], y=tmp["power_kw"])
axes[1, 0].set_title("Power by Weekday")
axes[1, 0].set_xticks(
    ticks=[0, 1, 2, 3, 4, 5, 6],
    labels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
)
axes[1, 0].set_xlabel("Weekday")
axes[1, 0].set_ylabel("Power (kW)")

sns.boxplot(ax=axes[1, 1], x=tmp["hour"], y=tmp["power_kw"])
axes[1, 1].set_title("Power by Hour of Day")
axes[1, 1].set_xlabel("Hour of Day")
axes[1, 1].set_ylabel("Power (kW)")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(presentation_output / POWER_ANALYSIS_PLOT)
plt.close(fig)

DATA_ANALYSIS_COMMENT = """
- Daily average usage appears non-stationary over the year, with higher consumption during summer.
- Large intra-day variations are observed, with low activity at night.
- Moderate variations are observed across weekdays, with higher usage on weekends.
- The distribution is right-skewed with a large mass at zero, corresponding to idle periods.
- Power consumption appears capped around ~700 kW, which is consistent with the station hardware limits.
""".strip()

# -----------------------------------------------------------------------------
# Modeling
# -----------------------------------------------------------------------------

power_by_15_minutes = (
    power_by_minute["power_kw"].resample("15min").mean().rename("avg_power_kw_15m")
)

# Hour and day_of_week capture daily/weekly seasonality
seasonal_components = pd.DataFrame(index=power_by_15_minutes.index).assign(
    hour=lambda df: df.index.hour.astype("category"),
    day_of_week=lambda df: df.index.dayofweek.astype("category"),
)

# Rolling statistics for slow level shifts.
# 14 days * 24 hours/day * 4 quarters/hour = 1344 steps
trend_component = RollingFeatures(
    stats="mean",
    window_sizes=14 * 24 * 4,
)

# 24 hours ahead at 15-minute resolution => 96 steps
STEPS_24H_15M = 24 * 4

estimator = LGBMRegressor(
    objective="tweedie",
    random_state=111,
    verbose=-1,
)

forecaster = ForecasterDirect(
    estimator=estimator, steps=STEPS_24H_15M, window_features=trend_component
)

# Baseline forecast: average of the last 10 observations
# taken from the equivalent day-of-week and 15-minute interval.
baseline_forecaster = ForecasterEquivalentDate(
    offset=24 * 4 * 7, n_offsets=10, agg_func=np.mean
)

MODELING_CHOICES_COMMENT = """
### Features
- Hour of day and day of week are used as exogenous variables to capture daily and weekly consumption patterns.
- A 14-day rolling mean of the 15-minute target series is used to model slow-moving level shifts and non-stationarity.

### Model
- Gradient boosting is used to capture non-linear effects and feature interactions (e.g. different behaviors on weekends).
- A Tweedie objective is well suited for a right-skewed target with many low or zero values.
- LightGBM hyperparameters are intentionally kept close to defaults for this proof-of-concept work.

### Baseline
- The baseline forecast is defined as the average of the last 10 observations from the equivalent day-of-week and time slot.

### Output
- A 96-step forecast (15-minute resolution over 24 hours) aligns with power billing rules based on 15-minute averages.

### Metrics
- MAE reflects the objective of minimizing energy procurement costs given known price curves.
- Maximum underestimation is critical, as missing demand peaks may force an increase in contracted maximum power.
""".strip()

# Backtesting (online-like): expanding window with regular refits
# -----------------------------------------------------------------------------
# The model is refit once per day (every 96 steps) and evaluated on rolling 24h-ahead forecasts

cv = TimeSeriesFold(
    steps=STEPS_24H_15M,
    initial_train_size=200 * 24 * 4,  # 200 days of 15-min data
    window_size=STEPS_24H_15M,  # advance by one day
    refit=True,
    fixed_train_size=False,
    gap=0,
    allow_incomplete_fold=False,
    verbose=False,
)


def maximal_underestimation_metric(y_true, y_pred):
    """
    Compute the maximum amount by which the forecast underestimates actual consumption.
    """
    return max(np.max(y_true - y_pred), 0.0)


metric, predictions = backtesting_forecaster(
    forecaster=forecaster,
    y=power_by_15_minutes,
    exog=seasonal_components,
    cv=cv,
    metric=["mean_absolute_error", maximal_underestimation_metric],
    verbose=True,
    show_progress=True,
)


metric_baseline, predictions_baseline = backtesting_forecaster(
    forecaster=baseline_forecaster,
    y=power_by_15_minutes,
    exog=seasonal_components,
    cv=cv,
    metric=["mean_absolute_error", maximal_underestimation_metric],
    verbose=True,
    show_progress=True,
)


# -----------------------------------------------------------------------------
# Results plot
# -----------------------------------------------------------------------------

MODELING_RESULT_PLOT = "forecasting_predictions_15m.png"

graph_index = predictions.index[-500:]

plt.figure(figsize=(12, 6))
plt.plot(power_by_15_minutes.loc[graph_index], label="Observed (15-min avg kW)")
plt.plot(predictions.loc[graph_index]["pred"], label="Forecast")
plt.plot(predictions_baseline.loc[graph_index]["pred"], label="Baseline Forecast")
plt.title("Forecasting: 15-min Resolution (1-step Ahead) vs Observed")
plt.xlabel("Timestamp")
plt.ylabel("Average power over 15 min (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(presentation_output / MODELING_RESULT_PLOT)
plt.close()


metrics_combined = pd.concat(
    [
        metric_baseline.assign(model="baseline"),
        metric.assign(model="lgbm"),
    ]
)[
    [
        "model",
        "mean_absolute_error",
        "maximal_underestimation_metric",
    ]
]

METRICS_TABLE = metrics_combined.to_markdown(index=False)

MODELING_RESULTS_COMMENT = """
### Model results
- The backtest MAE shows a clear performance improvement over the baseline.
- The model can occasionally severely underforecast consumption peaks, which may be problematic for the optimizer when trying to avoid increases in maximum contracted power.

### Next steps
- Incorporating yearly seasonality and calendar events would likely outperform a simple rolling mean once multiple years of data are available.
- The forecaster should output predictive distributions (e.g. quantiles or variance estimates) rather than point forecasts only, allowing the optimizer to explicitly handle uncertainty when planning energy purchases.
""".strip()

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

EXECUTIVE_SUMMARY = """
 This work presents a proof-of-concept online forecasting system for electricity demand at an electric vehicle charging station, with horizons ranging from 1 hour to 24 hours ahead. The objective is to provide reliable demand forecasts to support battery operation and energy procurement decisions, with a particular focus on limiting peak power consumption, which directly drives power-related costs.

Using minute-level power measurements, the data is aggregated to a 15-minute resolution to align with billing constraints. Exploratory analysis highlights strong intra-day seasonality, non-stationarity over the year, and a highly right-skewed distribution with frequent idle periods.

The forecasting approach combines a gradient boosting model (LightGBM) with calendar-based exogenous features and rolling statistics to capture both seasonal patterns and slow-moving level shifts. An online-like backtesting strategy with daily refits is used to simulate realistic production conditions. Model performance is evaluated against a simple historical baseline using both mean absolute error and a custom metric designed to penalize demand maximal underestimation.

Results show a clear improvement over the baseline in terms of average accuracy, while also highlighting remaining challenges in forecasting rare but impactful demand peaks. Potential next steps include incorporating longer-term seasonality, calendar effects, and probabilistic forecasts to better quantify uncertainty for downstream optimization.
"""

# -----------------------------------------------------------------------------
# Save artefacts
# -----------------------------------------------------------------------------

artefacts = {
    "executive_summary": EXECUTIVE_SUMMARY,
    "power_analysis_plot": POWER_ANALYSIS_PLOT,
    "modeling_result_plot": MODELING_RESULT_PLOT,
    "metrics_table": METRICS_TABLE,
    "data_analysis_comment": DATA_ANALYSIS_COMMENT,
    "modeling_choices_comment": MODELING_CHOICES_COMMENT,
    "modeling_results_comment": MODELING_RESULTS_COMMENT,
    "resolution": "15min",
    "horizon_steps": STEPS_24H_15M,
}

artefacts_path = presentation_output / "artefacts.json"
artefacts_path.write_text(json.dumps(artefacts, indent=2), encoding="utf-8")
