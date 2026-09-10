"""
Shared configuration: file paths and constant column definitions.
Keeping these in one place means every other module (data_prep, train,
predict) refers to the same paths and column lists, so nothing drifts.
"""
from pathlib import Path

# --- Directories -----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
EDA_PLOT_DIR = OUTPUT_DIR / "eda_plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EDA_PLOT_DIR.mkdir(parents=True, exist_ok=True)

# --- Input files -------------------------------------------------------------
TRAIN_TEST_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
VALIDATION_TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_INPUT_PATH = DATA_DIR / "december_chart_inputs.csv"

# --- Output files ------------------------------------------------------------
MODEL_PATH = MODEL_DIR / "freight_rate_model.joblib"
VALIDATION_PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions.csv"
DECEMBER_PREDICTIONS_PATH = OUTPUT_DIR / "december_chart_inputs_filled.csv"
METRICS_PATH = OUTPUT_DIR / "holdout_metrics.json"

# --- Columns -------------------------------------------------------------
TARGET_COL = "posted_rate"
ID_COL = "load_id"
DATE_COL = "date"

CATEGORICAL_COLS = ["equipment", "pickup", "delivery"]
BASE_NUMERIC_COLS = [
    "distance",
    "weight",
    "market_index",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]
# Engineered date features added by data_prep.add_date_features()
DATE_FEATURE_COLS = [
    "month",
    "day_of_week",
    "is_weekend",
    "day_of_year_sin",
    "day_of_year_cos",
]

FEATURE_COLS = BASE_NUMERIC_COLS + DATE_FEATURE_COLS + CATEGORICAL_COLS

RANDOM_STATE = 42
