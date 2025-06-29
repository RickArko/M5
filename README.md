# M5 Forecasting Challenge Solution
Using 5 years of Walmart Sales data in 10 stores, generate a 28-day forecast for `30,490 unique time-series`.

# Setup/Installation
This will pull another repo describing the challenge, install the basic dependencies and download and process the dataset. This will create two local directories inside the root:
  1. `data/`
  1. `DS-CaseStudy-SalesForecast`

These will both be added to the .gitignore by default.
```
    bash install.sh
```

# Exporatory Data Analysis
See `WriteUp.md` for a summary analysis, and run the `EDA.ipynb` notebook to reproduce the plots. The `NaiveForecast.ipynb` notebook implements a Naive baseline and the `Forecast.ipynb` notebook implements a few statistical baselines and a `LightGbm` regression forecast.

### References
Howard, A., Inversion, Makridakis, S., & Vangelis. (2020). *M5 Forecasting - Accuracy*. Kaggle. Retrieved from [https://kaggle.com/competitions/m5-forecasting-accuracy](https://kaggle.com/competitions/m5-forecasting-accuracy).
