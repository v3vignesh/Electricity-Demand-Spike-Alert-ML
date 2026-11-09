# VoltIQ — Electricity Demand Spike Alert Using ML
**Team 11 | Luxury Purple Streamlit Dashboard**

## Project contents
- `app.py` — complete Streamlit dashboard, model training, evaluation and prediction UI
- `electricity_demand.csv` — 1,399 hourly synthetic demonstration records
- `requirements.txt` — Python dependencies

## Run in VS Code (Windows)
1. Extract/open this project folder in VS Code.
2. Install Python 3.10 or newer.
3. Open **Terminal → New Terminal** in VS Code.
4. Create and activate a virtual environment (recommended):

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

5. Install packages:

   ```powershell
   python -m pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```powershell
   python -m streamlit run app.py
   ```

7. Your browser opens the VoltIQ dashboard. Keep the terminal running while using it.

## What the app does
- Validates the required dataset columns and binary target.
- Uses a chronological 80% train / 20% test split.
- Compares a majority-class baseline, Logistic Regression, Random Forest and HistGradientBoosting.
- Fits imputers/scalers only on training data through scikit-learn pipelines.
- Reports accuracy, precision, recall, F1 and ROC AUC, plus a confusion matrix.
- Shows two error slices: time period and weekday/weekend.
- Accepts validated new inputs and predicts next-hour spike probability.
- Saves the selected fitted pipeline as `best_spike_model.joblib` when the app runs.

## Important academic note
The included CSV is **synthetic demonstration data**, not official utility/consumer data. It has 1,399 hourly rows and more than 6 candidate predictors so you can run and demonstrate the interface. Your Cybernaut SRS requires you to propose three candidate datasets and obtain mentor approval for one legitimate dataset before final evaluation. Replace this CSV with the approved dataset, preserving the documented schema or update `FEATURES`, target logic and data dictionary in `app.py` accordingly. Record the source, licence/permission, collection dates, version and limitations in your report.

## Target definition
`spike_next_hour = 1` when the following hour's demand is greater than the current rolling 24-hour average by 48 MW; otherwise it is 0. This is a teaching definition for the synthetic demo, not a grid-operator threshold.

## Reproducibility
Random seeds are fixed at 42. The test set is the latest 20% of records in time order and is not used to fit preprocessing or model parameters. The app re-trains the models when it starts and writes the best non-baseline model to `best_spike_model.joblib`.
