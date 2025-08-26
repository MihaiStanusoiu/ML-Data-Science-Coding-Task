# Candidate Test

## Overview

This test is designed to evaluate your **Data Science** and **Deep Learning** skills.

You will work on a small photovoltaic (PV) generation forecasting task and have available the following datasets:

* **Historical generation** of the PV system
* **Weather measurements** of a nearby measurement station
* **Irradiation forecasts** for a nearby location

The goal is to prepare and select the data, and train a model to forecast PV generation for predictions 2 hours into the future.

---

## Project Requirements

You must complete the following tasks:

### 1. Set up a project

Set up a Python project folder and install your required dependencies.

---

### 2. Explore the data

Explore the data to get to know the contents and decide what to include into your model training.

What are good selection criteria to achieve a model that can be queried for a forecast at present time?

Use statistical analysis and plotting options for this step.

---

### 3. Prepare the data

What data preparation steps are necessary for the model training? Do them.

---

### 4. Train a model

Decide on a suitable neural network based algorithm for time series forecasting and train a simple model.

The choice of model will be evaluated in terms of the suitability for the time series forecasting task and your understanding of the architecture.

Check how well your model converged.

---

### 5. Evaluate the training and the model

Evaluate the model on a test dataset and plot the model predicitons for 2 hours into the future. In the same plot compare with the measured values.

What metric(s) could be used to quantify the results?

Think about how the model could be further improved.

---

### Evaluation Criteria

Your presentation will be evaluated based on:

1. **Steps taken to prepare the dataset**

   * Data exploration and understanding
   * Data selection and preparation for training

2. **Learning algorithm**

   * Choice and basic implementation of a suitable neural network based time series forecasting algorithm
   * Understanding of the algorithm

3. **Model training**

   * Knowledge of influential hyperparameters and options to tune them
   * Insights on model convergence or non-convergence in the training process

4. **Forecasting**
  
    * Clear visualisation of forecasting results
    * Ideas and thoughts how to quantify the results with metrics and how the model could be improved in the future

5. **Code Quality**

   * Clean, modular code
   * Meaningful variable and function names
  
6. **Documentation**

   * Clear documentation of code, steps taken and results
   * Understandable visualizations

---

### Dataset description

**pv_generation.csv**: Contains hourly PV power measurement data in kW.

**weather_measurements.csv**: Contains hourly weather measurements. The temporal index is in UTC. The meanings of the columns are: 'station' - a weather station id, 'cglo' - global horizontal irradiation in W/m2, 'ff' - wind speed in m/s, 'tl' - air temperature in degree Celcius.

**data_ghi_forecast.parquet**: Contains hourly global horizontal irradiation forecast data. The index is a pandas.MultiIndex with the first entry in the tuple being the valid time and the second the transaction time as a time offset compared to the valid time. I.e. (Timestamp('2022-10-15 00:00:00+0000', tz='UTC'), Timedelta('0 days 04:00:00')) means the forecast is valid at midnight of the 15th and was created 4 hours earlier. The unit can be considered to be arbitrary.