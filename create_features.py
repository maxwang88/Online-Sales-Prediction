"""Create features from csv files"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
from tqdm import tqdm


def create_features(df):
    """Conduct feature engineering on raw data.
    Args:
        df: Dataframe with "quan", "cat", "date" variables only. No sales information.
    Returns:
        df_vari: Dataframe contains cleaned data without sales information.
    """

    # identify categorical variables and quant variables
    cat_cols = [i for i in df.columns if 'Cat' in i]

    quan_log_cols = ["Quan_4", "Quan_5", "Quan_6", "Quan_7", "Quan_8", "Quan_9",
                     "Quan_10", "Quan_11", "Quan_12", "Quan_13", "Quan_14", "Quan_15",
                     "Quan_16", "Quan_19", "Quan_27", "Quan_28", "Quan_29", "Quant_22",
                     "Quant_24", "Quant_25"]

    # quan to cat
    quan2cat_cols = ["Quan_17", "Quan_18", "Quan_21", "Quan_22", "Quant_23", "Quan_26",
                     "Quan_27", "Quan_28", "Quan_29", "Quan_30"]

    quan_cols = set([i for i in df.columns if 'Quan' in i and i not in quan_log_cols + quan2cat_cols])

    df.loc[:, quan_log_cols].fillna(df[quan_log_cols].median(), inplace=True)
    df.loc[:, quan2cat_cols] = df.loc[:, quan2cat_cols].astype(str)

    df_vari = pd.DataFrame()

    # transfer date to year, month, day
    for col in ['Date_1', 'Date_2']:
        dt = df[col].apply(lambda x: datetime.date(99, 1, 1) if np.isnan(x) else datetime.date.fromordinal(int(x)))
        df[col + '_Cat_year'] = dt.apply(lambda x: 0 if x == datetime.date(99, 1, 1) else x.year)
        df[col + '_Cat_month'] = dt.apply(lambda x: 0 if x == datetime.date(99, 1, 1) else x.month)
        # df[col + '_Cat_day'] = dt.apply(lambda x: 0 if x == datetime.date(99, 1, 1) else x.day)

    for col in df.columns:
        if "Outcome" in col:
            continue
        # if the value of this column is invariance, do not add this column to feature
        if (df[col].nunique() == 1 and df[col].isnull().sum() == 0) or df[col].nunique() == 0:
            continue
        # if it's a categorical column
        if 'Cat' in col or col in quan2cat_cols:
            if (df[col].nunique() > 1) or (df[col].isnull().sum() > 0 and df[col].nunique() > 0):
                for val in df[col].unique():
                    df_vari[col + '_' + str(val)] = (df[col] == val).astype(int)
                if df[col].isnull().sum() > 0:
                    df_vari[col + '_NaN'] = df[col].isnull().astype(int)
        elif col in quan_log_cols:
            df[col].fillna(df[col].median(), inplace=True)  # fill nan with median
            vals = np.log1p(df[col].values)
            df_vari[col] = vals
        elif col in quan_cols:
            df[col].fillna(df[col].median(), inplace=True)  # fill nan with median
            df_vari[col] = df[col].values

    df_vari['Date_1'] = df['Date_1'].values
    df_vari['Date_2'] = df['Date_2'].values
    df_vari['Date_2'].fillna(0, inplace=True)
    df_vari['time_diff'] = df['Date_1'] - df['Date_2']
    df_vari.loc[df_vari['time_diff'].isnull(), 'time_diff'] = 0

    duplicates = []
    X = df_vari.values
    print('Removing same features ...')
    for i in tqdm(range(X.shape[1] - 1)):
        for j in range(i + 1, X.shape[1]):
            if (X[:, i] == X[:, j]).all():
                duplicates.append(j)
    duplicates = set(duplicates)

    df_vari = df_vari.iloc[:, [i for i in range(df_vari.shape[1]) if i not in duplicates]].copy()

    return df_vari


if __name__ == "__main__":
    # read data from csv
    df_train = pd.read_csv('TrainingDataset.csv')
    df_test = pd.read_csv('TestDataset.csv')
    del df_test['id']

    # remove the target variables from feature processing
    df = pd.concat([df_train.iloc[:, 12:], df_test])
    df_vari = create_features(df)
    data_train = pd.concat([df_train.iloc[:,:12], df_vari.iloc[:df_train.shape[0], :]], axis=1)
    # save cleaned data for training
    data_train.to_csv('clean_data_train.csv', index=False)

    data_test = df_vari.iloc[df_train.shape[0]:, :].copy()
    # save cleaned data for predicting and submission
    data_test.to_csv('clean_data_test.csv', index=False)