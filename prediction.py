"""Make predictions using the model created in "create_model.py"."""

import pandas as pd
import numpy as np
import pickle


from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def multi_month_2_one(df_X, df_y):
    X_con = []
    for col in df_y.columns:
        bools = ~df_y[col].isnull()
        df_buf = df_X.loc[bools, :].copy()
        df_buf['month'] = int(col.split('M')[1])
        # one_hot_encode ''month''
        for i in range(1, 13):
            df_buf['month_' + str(i)] = (df_buf['month'] == i).astype(int)
        del df_buf['month']

        X_con.append(df_buf)
    X = pd.concat(X_con)

    return X


if __name__ == "__main__":
    loaded_model = pickle.load(open("xgb.dat", "rb"))
    df = pd.read_csv('clean_data_test.csv')
    df_train = pd.read_csv('TrainingDataset.csv')
    df['idx'] = range(df_train.shape[0],df_train.shape[0] + df.shape[0])
    month_cols = df_train.columns[:12]
    df_y = pd.DataFrame(columns=month_cols, data=np.zeros([df.shape[0], 12]))

    df_X = multi_month_2_one(df, df_y)

    y_pred = loaded_model.predict(df_X.values)

    # transfer log to original sales amount
    y_pred = np.exp((y_pred - 1))

    df_sub = pd.read_csv('sample_submission.csv')
    for i in range(1, 13):
        df_sub.iloc[:, i] = y_pred[(i - 1) * df_sub.shape[0]:i * df_sub.shape[0]]

    df_sub.to_csv('submission.csv', index=False)
