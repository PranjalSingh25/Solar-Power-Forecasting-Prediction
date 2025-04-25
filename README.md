# Solar Power Forecasting Pipeline (LSTM + pvlib) ☀️

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg) <!-- Choose a license, MIT is common -->

This project demonstrates an end-to-end pipeline for predicting hourly solar photovoltaic (PV) power output using historical weather data, physics-based simulation, and a Long Short-Term Memory (LSTM) deep learning model.

## Project Overview

This project aims to forecast the AC power generation of a defined solar PV system for future time steps (initially 1 hour ahead via API, extendable to multiple days with a weather forecast). It leverages:

1.  **Data Acquisition:** Fetches historical hourly weather data (including Global Horizontal, Direct Normal, and Diffuse Horizontal Irradiance, Temperature, and Wind Speed) from the NASA POWER API.
2.  **PV Simulation:** Utilises the `pvlib-python` library to simulate a specific, user-configured PV system's theoretical hourly AC power output (Watts) based on the fetched weather data. This serves as the target variable for training.
3.  **LSTM Model Training** Trains a PyTorch-based LSTM model to learn the relationship between the sequence of past hourly weather features and the corresponding simulated power output. Includes proper data splitting, scaling, and evaluation.
4.  **API Deployment:** Deploys the trained LSTM model using a Flask web service, providing a REST API endpoint (`/predict`) to get predictions for the next hour based on the last sequence of weather data.
5.  **Multi-Day Forecasting (Optional Script):** Includes a separate script (`forecast_5day.py`) demonstrating how to use the trained model iteratively with an external weather *forecast* to predict power output over multiple days.

## Features ✨

*   **Automated Data Fetching:** Downloads required hourly weather parameters from NASA POWER API for any specified location and date range (respecting API limits).
*   **Physics-Based Simulation:** Simulates realistic PV power output using `pvlib`, configurable for different system parameters (location, panel/inverter models, tilt, azimuth, etc.).
*   **LSTM Time Series Forecasting:** Implements an LSTM model in PyTorch for sequence-to-value prediction.
*   **Standard ML Workflow:** Includes data preprocessing (handling NaNs, scaling), chronological train/validation/test splitting, model training with early stopping, and robust evaluation (RMSE, MAE, R²).
*   **Model Serving API:** The Flask application exposes the trained model via a simple REST API for single-step-ahead predictions.
*   **Multi-Step Forecasting Capability:** Demonstrates iterative forecasting using a separate script (requires external weather forecast input).
*   **Visualisation:** Generates plots for training history and prediction results (test set & multi-day forecast).
*   **Modular Code:** Organised into distinct Python scripts for each stage of the pipeline.

## Technologies Used 🛠️

*   **Language:** Python (3.8+)
*   **Deep Learning:** PyTorch
*   **PV Simulation:** pvlib-python
*   **Web Framework / API:** Flask
*   **Data Handling:** Pandas, NumPy
*   **Machine Learning Utilities:** Scikit-learn (for scaling, metrics), Joblib (for saving scalers)
*   **API Interaction:** Requests
*   **Plotting:** Matplotlib
*   **Utilities:** Pathlib, Logging, Argparse, TimezoneFinder (optional, for timezone lookup)

## Setup & Installation ⚙️

1.  **Clone the Repository:**
    ```bash
    git clone [[Your Repository URL](https://github.com/PranjalSingh25/Solar-Power-Forecasting-Prediction)]
    cd [Solar-Power-Forecasting-Prediction]
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Requirements:**
    *   **Create `requirements.txt`:** Make sure you have a `requirements.txt` file listing all necessary libraries. You can generate one after installing it manually:
        ```bash
        pip install torch pandas numpy scikit-learn pvlib flask requests joblib matplotlib timezonefinder # Add pyarrow/fastparquet if using parquet
        pip freeze > requirements.txt
        ```
    *   **Install from the file:**
        ```bash
        pip install -r requirements.txt
        ```

## Usage 🚀

Follow these steps in order:

**1. Fetch Hourly Weather Data:**

*   Run the data fetching script. It will prompt you for Latitude, Longitude, Start Date, and End Date.
    ```bash
    python solar_data_pipeline_2.py
    ```
*   This creates `data/nasa_power_hourly_raw.csv` (and potentially `.parquet`).
*   *Note:* Due to API limits, you may need to run this multiple times for different date ranges to get several years of data and concatenate the resulting CSVS manually if desired.

**2. Simulate PV Power:**

*   **‼️ IMPORTANT:** Before running, **edit the `simulate_pv_power.py` script** and update the `--- Configuration ---` section (Lines 15-54) with your specific location (must match fetch step!) and detailed PV system parameters (panel model, inverter model, tilt, azimuth, layout). Accurate parameters are crucial for meaningful results.
*   Run the simulation script:
    ```bash
    python simulate_pv_power.py
    ```
*   This reads the raw hourly weather data and creates `data/processed/weather_and_simulated_hourly_power.csv`, which includes the calculated `simulated_ac_power_w` column.

**3. Train the LSTM Model:**

*   Run the training script. You can use default hyperparameters or override them via command-line arguments (see `python train_lstm.py -h`).
    ```bash
    python train_lstm.py
    ```
    *Example with custom arguments:*
    ```bash
    python train_lstm.py --seq_len 48 --epochs 100 --lr 0.0005
    ```
*   This script trains the model, performs validation and testing, and saves the following in the `models/` directory:
    *   `best_lstm_model_hourly.pth` (Trained model weights)
    *   `feature_scaler_hourly.joblib` (Scaler for input features)
    *   `target_scaler_hourly.joblib` (Scaler for the power output)
*   It also saves plots (`training_history_hourly.png`, `test_predictions_hourly.png`) in the `plots/` directory and logs detailed metrics.

**4. Run the Prediction API:**

*   Start the Flask server. This will typically run in the foreground in your terminal.
    ```bash
    python app.py
    ```
*   The API will be available at `http://127.0.0.1:5001`.
*   You can now send `POST` requests to the `/predict` endpoint (see details below).
*   Press `Ctrl+c` in the terminal to stop the server.

**5. (Optional) Generate a 5-Day Forecast:**

*   **‼️ PREREQUISITE:** You **must** first obtain an hourly weather *forecast* for the next 5 days (120 hours) for your location. This forecast needs to include the same columns used for training (`ALLSKY_SFC_SW_DWN`, `ALLSKY_SFC_SW_DNI`, etc.). Save this forecast as a CSV file named `weather_forecast_next_5_days.csv` inside the `forecasts/` directory. The CSV needs a 'Timestamp' column/index.
*   Ensure the configuration (especially `SEQUENCE_LENGTH` and `FEATURE_COLS`) in `forecast_5day.py` matches the training script.
*   Run the forecasting script:
    ```bash
    python forecast_5day.py
    ```
*   This will generate `plots/power_forecast_5day.png` and optionally save the numerical forecast data to `forecasts/power_forecast_results_5day.csv`.

## API Endpoint (`/predict`) Details

*   **URL:** `http://127.0.0.1:5001/predict`
*   **Method:** `POST`
*   **Headers:** `Content-Type: application/json`
*   **Body (Raw JSON):** Needs to contain a key `"data"` whose value is a list of dictionaries. This list must contain **exactly `SEQUENCE_LENGTH`** (e.g., 24) consecutive hourly weather data points, ordered chronologically with the most recent hour last. Each dictionary needs keys matching the `FEATURE_COLS` used during training.

    ```json
    {
      "data": [
        { // Hour -23 (Oldest)
          "ALLSKY_SFC_SW_DWN": ...,
          "ALLSKY_SFC_SW_DNI": ...,
          "ALLSKY_SFC_SW_DIFF": ...,
          "T2M": ...,
          "WS10M": ...
        },
        // ... more dictionaries ...
        { // Hour -1 (Most Recent)
          "ALLSKY_SFC_SW_DWN": ...,
          "ALLSKY_SFC_SW_DNI": ...,
          "ALLSKY_SFC_SW_DIFF": ...,
          "T2M": ...,
          "WS10M": ...
        }
      ]
    }
    ```

*   **Example `curl` Request:**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d '{ "data": [ { "ALLSKY_SFC_SW_DWN": ... }, ... 23 more ... , { "ALLSKY_SFC_SW_DWN": ... } ] }' http://127.0.0.1:5001/predict
    ```
    *(Replace the body with your actual valid JSON data)*

*   **Success Response (200 OK):**
    ```json
    {
      "predicted_power_w": 1234.56
    }
    ```
*   **Error Responses:** Returns JSON with an "error" key and appropriate HTTP status codes (400 for bad input, 500 for server errors, 503 if service is not ready).

## Results 📊

The LSTM model trained on the simulated data achieved the following performance on the test set:

*   **RMSE:** [192.38 W]
*   **MAE:** [ 106.78 W]
*   **R²:** [0.9816]

See `plots/training_history_hourly.png` for loss curves and `plots/test_predictions_hourly.png` for a visualisation of test set predictions versus simulated actuals. The `plots/power_forecast_5day.png` shows an example multi-day forecast output (dependent on the quality of the input weather forecast).

## Future Improvements 🚀

*   **Integrate Actual Measured Data:** Train/evaluate the model using real-world measured power output data (if available) instead of just simulated data for potentially higher real-world accuracy. Implement residual modelling (predicting the difference between simulation and actual).
*   **Hyperparameter Tuning:** Use techniques like grid search, random search, or Bayesian optimisation to find optimal LSTM parameters (hidden size, layers, dropout, sequence length, learning rate).
*   **Feature Engineering:** Add more relevant features, such as sine/cosine transformations of hour-of-day and day-of-year, cloud cover forecasts, or lagged power output values.
*   **Explore Other Models:** Compare LSTM performance against other time series models like GRUS, Transformers, or even simpler models like Random Forest or Gradient Boosting (with appropriate feature engineering).
*   **Robust Weather Forecasting:** Integrate a reliable weather forecast API directly into the `forecast_5day.py` script instead of relying on a manual CSV file. Handle potential discrepancies between forecast parameters and training parameters.
*   **Cloud Deployment:** Deploy the Flask API to a cloud platform (like AWS, Google Cloud, Azure, or Heroku) for public accessibility.
*   **Error Analysis:** Deeper dive into *when* the model makes the largest errors (e.g., specific weather conditions, time of day).


## Contact 📫

[Pranjal Singh] - [https://www.linkedin.com/in/pranjal-singh-265937286/]

Project Link: [https://github.com/PranjalSingh25/Solar-Power-Forecasting-Prediction](https://github.com/PranjalSingh25/Solar-Power-Forecasting-Prediction)
