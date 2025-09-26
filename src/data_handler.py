import pandas as pd
import io

def load_dataset_from_filelike(file):
    file.seek(0)
    content = file.read()
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception:
        try:
            return pd.read_csv(io.BytesIO(content), sep=';')
        except Exception:
            return pd.read_excel(io.BytesIO(content))

def load_dataset(path):
    return pd.read_csv(path)
