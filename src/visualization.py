import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns

def plot_confusion(cm):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

def plot_price_with_preds(dates, prices, up_idx, down_idx):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(dates, prices, label='Price')
    if up_idx:
        ax.scatter([dates[i] for i in up_idx], [prices[i] for i in up_idx], color='green', marker='^', label='Pred Up')
    if down_idx:
        ax.scatter([dates[i] for i in down_idx], [prices[i] for i in down_idx], color='red', marker='v', label='Pred Down')
    ax.legend()
    st.pyplot(fig)

def plot_forecast(dates_hist, prices_hist, dates_future, prices_future):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(dates_hist, prices_hist, label='History')
    ax.plot(dates_future, prices_future, label='Forecast', color='orange')
    ax.fill_between(dates_future, [p*0.98 for p in prices_future], [p*1.02 for p in prices_future], alpha=0.2, color='orange')
    ax.legend()
    st.pyplot(fig)
