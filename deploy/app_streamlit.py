\
import streamlit as st
import pandas as pd
import numpy as np
import os, sys, subprocess, io
from pathlib import Path

# ensure project root is importable
sys.path.append(str(Path(__file__).resolve().parents[1]))

from deploy import config
from src import data_handler, preprocessing, model, visualization
from utils.helpers import format_prediction
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

st.set_page_config(page_title=config.APP_NAME, page_icon="📈", layout="wide")
st.title(config.APP_NAME)

# Sidebar: demo mode and dataset selection
demo_mode = st.sidebar.checkbox("Demo mode (load Nifty sample)", value=True)
choice = st.sidebar.radio("Load data", ["Nifty 50", "BankNifty", "Upload your dataset"])

df = None
if demo_mode:
    # demo forces Nifty regardless
    df = data_handler.load_dataset(str(Path(__file__).resolve().parents[1] / "data" / "nifty50.csv"))
else:
    if choice == "Nifty 50":
        df = data_handler.load_dataset(str(Path(__file__).resolve().parents[1] / "data" / "nifty50.csv"))
    elif choice == "BankNifty":
        df = data_handler.load_dataset(str(Path(__file__).resolve().parents[1] / "data" / "banknifty.csv"))
    else:
        uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv","xls","xlsx","txt"])
        if uploaded is not None:
            try:
                df = data_handler.load_dataset_from_filelike(uploaded)
            except Exception as e:
                st.error(f"Failed to read uploaded file: {e}")

if df is None or df.empty:
    st.error("Dataset is empty or not loaded. Use Demo mode or upload a valid CSV with numeric price columns (Open/High/Low/Close).")
    st.stop()

st.write("### Dataset preview")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head())

# select numeric price columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# prefer 'Close' if present
price_col = "Close" if "Close" in df.columns else (numeric_cols[0] if numeric_cols else None)
if price_col is None:
    st.error("No numeric columns found to act as price. Upload a dataset with a numeric price column.")
    st.stop()

st.write(f"Using **{price_col}** as price column for predictions and forecast.")

# prepare
df_prepared = preprocessing.prepare_stock_df(df, price_col)
if df_prepared is None or df_prepared.empty:
    st.error("After preprocessing the dataset is empty or invalid. Please check your file.")
    st.stop()

# create target (Up next day)
df_prepared["TARGET_UP"] = (df_prepared[price_col].shift(-1) > df_prepared[price_col]).astype(int)
df_prepared = df_prepared.dropna().reset_index(drop=True)
if df_prepared.empty or len(df_prepared) < 10:
    st.error("Not enough data after preprocessing to train models (need at least 10 rows). Please upload a larger dataset or disable demo mode.")
    st.stop()

# split (time-aware)
split_idx = int(0.8 * len(df_prepared))
train = df_prepared.iloc[:split_idx]
test = df_prepared.iloc[split_idx:]

X_train = train[[price_col]].values
y_train = train["TARGET_UP"].values
X_test = test[[price_col]].values
y_test = test["TARGET_UP"].values

# train classifier
clf = model.train_classifier(X_train, y_train, n_estimators=200)
y_pred = clf.predict(X_test)
y_proba = None
try:
    y_proba = clf.predict_proba(X_test)[:,1]
except Exception:
    y_proba = None

acc = accuracy_score(y_test, y_pred)
st.metric("Up/Down accuracy", f"{acc:.3f}")

# build results table with labels
results = pd.DataFrame({
    "Date": test["__DATE__"].dt.strftime("%Y-%m-%d").values if "__DATE__" in test.columns else test.index,
    "Price": X_test.flatten(),
    "Actual": ["📈 UP" if v==1 else "📉 DOWN" for v in y_test],
    "Predicted": ["📈 UP" if v==1 else "📉 DOWN" for v in y_pred]
})
st.subheader("Predictions (Up=1, Down=0)")
st.dataframe(results.head(50))

# summary
trend = "UP 📈" if np.mean(y_pred) >= 0.5 else "DOWN 📉"
if y_proba is not None:
    avg_conf = float(np.mean(y_proba))
    st.success(f"Model trend summary: predicts **{trend}** (avg up probability {avg_conf:.2f})")
else:
    st.success(f"Model trend summary: predicts **{trend}**")

# confusion matrix & visualization
cm = confusion_matrix(y_test, y_pred)
st.write("Confusion Matrix")
visualization.plot_confusion(cm)

# distribution
st.write("Prediction Distribution")
st.bar_chart(pd.Series(y_pred).value_counts().sort_index())

# price chart with preds markers
up_idx = [i for i,p in enumerate(y_pred) if p==1]
down_idx = [i for i,p in enumerate(y_pred) if p==0]
visualization.plot_price_with_preds(test["__DATE__"].values if "__DATE__" in test.columns else test.index, X_test.flatten(), up_idx, down_idx)

# 30-day forecast using simple lag-based regressor
st.markdown('---')
st.header("30-day Forecast (ML-based)")
# build lag features
n_lags = 5
df_reg = df_prepared[[price_col]].copy().reset_index(drop=True)
for lag in range(1, n_lags+1):
    df_reg[f"lag_{lag}"] = df_reg[price_col].shift(lag)
df_reg = df_reg.dropna().reset_index(drop=True)
feat_cols = [f"lag_{lag}" for lag in range(1, n_lags+1)]
Xr = df_reg[feat_cols].values
yr = df_reg[price_col].values
split_r = int(0.8 * len(Xr))
Xr_train, yr_train = Xr[:split_r], yr[:split_r]

if Xr_train.shape[0] == 0:
    st.error("Not enough data to train regression model. Please provide a larger dataset.")
    st.stop()

reg = model.train_regressor(Xr_train, yr_train, n_estimators=200)
last_window = df_reg[feat_cols].iloc[-1].values.tolist()
preds_30 = []
cur = last_window.copy()
for _ in range(config.FORECAST_DAYS):
    p = reg.predict([cur])[0]
    preds_30.append(p)
    cur = cur[1:] + [p]

# plot forecast
hist_dates = df_prepared["__DATE__"].values
hist_prices = df_prepared[price_col].values
future_dates = pd.date_range(start=pd.to_datetime(hist_dates[-1]) + pd.Timedelta(days=1), periods=len(preds_30))
visualization.plot_forecast(hist_dates[-100:], hist_prices[-100:], future_dates, preds_30)

# Inference mode (upload new file and get predictions using trained classifier)
st.markdown('---')
st.header("Inference Mode")
inf_file = st.file_uploader("Upload new CSV/XLSX for prediction (optional)", type=['csv','xls','xlsx'])
if inf_file is not None:
    try:
        new_df = data_handler.load_dataset_from_filelike(inf_file)
        if price_col not in new_df.columns:
            cand = new_df.select_dtypes(include=[np.number]).columns.tolist()
            if not cand:
                st.error("No numeric column to predict on in uploaded file.")
            else:
                col = cand[0]
                st.warning(f"Using detected numeric column '{col}' for prediction.")
        else:
            col = price_col
        new_df[col] = pd.to_numeric(new_df[col], errors='coerce').ffill().bfill().fillna(0)
        preds_new = clf.predict(new_df[[col]].values)
        new_df['Predicted_Up'] = preds_new
        new_df['Pred_Label'] = new_df['Predicted_Up'].apply(lambda v: '📈 UP' if v==1 else '📉 DOWN')
        st.write(new_df.head(50))
        csv = new_df.to_csv(index=False).encode()
        st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"Inference failed: {e}")

# Backend Kaggle (available but not shown in primary UI)
def _kaggle_download(slug, target_dir='data'):
    try:
        os.makedirs(target_dir, exist_ok=True)
        cmd = f"kaggle datasets download -d {slug} -p {target_dir} --unzip"
        res = subprocess.run(cmd.split(), capture_output=True, text=True, check=True)
        return True, res.stdout
    except Exception as e:
        return False, str(e)
