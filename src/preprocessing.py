import pandas as pd
import numpy as np

def prepare_stock_df(df, price_col):
    df = df.copy()
    # detect date-like col and set
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if date_cols:
        df['__DATE__'] = pd.to_datetime(df[date_cols[0]], errors='coerce')
    else:
        df['__DATE__'] = pd.RangeIndex(start=0, stop=len(df))
    # ensure numeric price col
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
    # forward/backward fill (avoid deprecated fillna with method)
    df = df.ffill().bfill().fillna(0)
    return df
