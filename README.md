# German Car Price Predictor

## Project files

Keep these files in one folder:

- `app.py`
- `car_price_pipeline.joblib`
- `cleaned_cars.csv`
- `requirements.txt`

The model file is the original saved Tuned Ridge Regression pipeline. The application does not retrain it.

## Dataset preparation

The supplied ZIP contained `autoscout24-germany-dataset.csv`, which is the original 46,405-row dataset rather than a pre-cleaned CSV. `cleaned_cars.csv` was reconstructed from the supplied dataset and the cleaning operations documented in `Predictions_model.ipynb`:

1. Remove exact duplicate rows.
2. Mark mileage values >= 900,000 as missing.
3. Mark horsepower values equal to 1 as missing.
4. Exclude makes `Trailer-Anhänger`, `Trucks-Lkw`, and `DAF`.

This produces 44,260 rows. Missing values are intentionally retained for the saved pipeline's preprocessing steps.

## Setup in Windows PowerShell

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

## Run

```powershell
streamlit run app.py
```

## Compatibility

The saved pipeline was created with scikit-learn 1.6.1 and joblib 1.6.0. Keep these versions pinned. If the model cannot be unpickled, verify:

```powershell
python -c "import sklearn, joblib; print(sklearn.__version__, joblib.__version__)"
```

Expected output: `1.6.1 1.6.0`.

## Notes

- Manufacturer/model choices are derived from the recreated training partition using the same 80/20 split and `random_state=42`.
- Missing model categories are represented as `Unknown`, matching the saved model's model-column imputer.
- Dataset mean and median are calculated from the recreated training target partition, not manually entered.
- The gauge's upper bound is based on the training target's 95th percentile and expands when a prediction exceeds it.
- The hero background uses an externally hosted Unsplash image; an internet connection is needed to display that image. The rest of the application works locally.
