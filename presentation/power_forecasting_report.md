# Power Demand Forecasting – Proof of Concept

## Executive Summary


 This work presents a proof-of-concept online forecasting system for electricity demand at an electric vehicle charging station, with horizons ranging from 1 hour to 24 hours ahead. The objective is to provide reliable demand forecasts to support battery operation and energy procurement decisions, with a particular focus on limiting peak power consumption, which directly drives power-related costs.

Using minute-level power measurements, the data is aggregated to a 15-minute resolution to align with billing constraints. Exploratory analysis highlights strong intra-day seasonality, non-stationarity over the year, and a highly right-skewed distribution with frequent idle periods.

The forecasting approach combines a gradient boosting model (LightGBM) with calendar-based exogenous features and rolling statistics to capture both seasonal patterns and slow-moving level shifts. An online-like backtesting strategy with daily refits is used to simulate realistic production conditions. Model performance is evaluated against a simple historical baseline using both mean absolute error and a custom metric designed to penalize demand maximal underestimation.

Results show a clear improvement over the baseline in terms of average accuracy, while also highlighting remaining challenges in forecasting rare but impactful demand peaks. Potential next steps include incorporating longer-term seasonality, calendar effects, and probabilistic forecasts to better quantify uncertainty for downstream optimization.


## 1. Exploratory Analysis

![Power consumption analysis](power_analysis_plots.png)

- Daily average usage appears non-stationary over the year, with higher consumption during summer.
- Large intra-day variations are observed, with low activity at night.
- Moderate variations are observed across weekdays, with higher usage on weekends.
- The distribution is right-skewed with a large mass at zero, corresponding to idle periods.
- Power consumption appears capped around ~700 kW, which is consistent with the station hardware limits.

## 2. Modeling Choices

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


## 3. Backtesting Results

### Quantitative Metrics

| model    |   mean_absolute_error |   maximal_underestimation_metric |
|:---------|----------------------:|---------------------------------:|
| baseline |              122.534  |                          509.264 |
| lgbm     |               94.6819 |                          409.024 |

### Forecast vs Observations

![Forecasting results](forecasting_predictions_15m.png)

### Model results
- The backtest MAE shows a clear performance improvement over the baseline.
- The model can occasionally severely underforecast consumption peaks, which may be problematic for the optimizer when trying to avoid increases in maximum contracted power.

### Next steps
- Incorporating yearly seasonality and calendar events would likely outperform a simple rolling mean once multiple years of data are available.
- The forecaster should output predictive distributions (e.g. quantiles or variance estimates) rather than point forecasts only, allowing the optimizer to explicitly handle uncertainty when planning energy purchases.