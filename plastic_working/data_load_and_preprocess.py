import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats as stats
from warnings import filterwarnings
filterwarnings('ignore')

DROP_COLS = ['date', 'EX1.MELT_P_PV', 'EX1.MD_PV', 'EX1.MD_TQ']
TARGET = "passorfail"
ALPHA = 0.1

# 1-1. load data
def load_data(path, encoding="utf-8-sig"):
    return pd.read_csv(path, encoding=encoding)

def basic_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df1 = df.copy()
    cols = [c for c in df1.columns if c not in DROP_COLS]
    df2 = df1.loc[:, cols]

    # 타입 캐스팅
    df2 = df2.astype(float)
    df2 = df2.astype({TARGET: 'float32'})

    # 결측 제거
    df2 = df2.dropna()

    # 이상치 클리핑(0.1% ~ 99.9%)
    for col in df2.columns:
        if col == TARGET: 
            continue
        ub = np.percentile(df2[col], 99.9)
        lb = np.percentile(df2[col], 0.1)
        df2[col] = df2[col].clip(lb, ub)
    return df2

def select_features_ttest(df2: pd.DataFrame, target: str = TARGET, alpha: float = ALPHA):
    feats = [c for c in df2.columns]
    rows = []
    for col in feats:
        t = stats.ttest_ind(
            df2[df2[target] == 1][col],
            df2[df2[target] == 0][col],
            equal_var=False, nan_policy="omit"
        )
        rows.append((col, t.statistic, t.pvalue))
    df_t = pd.DataFrame(rows, columns=["col", "tvalue", "pvalue"])
    selected = df_t.loc[df_t["pvalue"] <= alpha, "col"].tolist()
    df3 = df2.loc[:, selected]

    return df3
