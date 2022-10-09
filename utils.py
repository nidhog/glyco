import pandas as pd
import enum

# Map for the weekday number and the name of the weekday
weekday_map = {0: 'Mon', 1: 'Tue', 2: 'Wed',
               3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}

# Utility function that returns True if a weekday number refers to a weekend
def is_weekend(x): return 1 if x % 7 > 4 else 0

class Devices(enum.Enum):
    abbott = "abbott"  # FreeStyle Libre

class Units(enum.Enum):
    mmol = "mmol/L"
    mgdl = "mg/dL"
    gl = "g/L"

def find_nearest(df, pivot, col, n_iter=100):
    items = list(df.index)
    n = items.copy()
    for i in range(n_iter):
        m = min(n, key=lambda x: abs(x - pivot))
        q=df.loc[m][col]
        if type(q)==pd.Series:
            q=q[0]
        if pd.isna(q):
            n.remove(m)
        else:
            return m