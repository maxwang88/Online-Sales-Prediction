"""Create model based on the parameters found in "modeling.html" """


import pandas as pd
import numpy as np
import pickle

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def multi_month_2_one(df_X, df_y):
    """If use the single model approach, the month of each sales
    should be transferred to a feature variable."""
    y_con = []
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
        y_con.extend(list(np.log1p(df_y.loc[bools, col].values)))
    X = pd.concat(X_con)
    y = np.array(y_con)
    return X, y


if __name__ == "__main__":
    df = pd.read_csv('clean_data_train.csv')
    df['idx'] = range(df.shape[0])

    df_X = df.iloc[:, 12:].copy()
    df_y = df.iloc[:, :12].copy()

    test_size = 0.3
    seed = 7
    df_X_train, df_X_test, df_y_train, df_y_test = train_test_split(df_X, df_y, test_size=test_size,
                                                                    random_state=seed)

    df_X_train_train, df_X_train_validate, df_y_train_train, df_y_train_validate =\
    train_test_split(df_X_train, df_y_train, test_size=test_size, random_state=seed)

    X_train_validate, y_train_validate = multi_month_2_one(df_X_train_validate, df_y_train_validate)
    X_train_train, y_train_train = multi_month_2_one(df_X_train_train, df_y_train_train)
    X_test, y_test = multi_month_2_one(df_X_test, df_y_test)
    X_train, y_train = multi_month_2_one(df_X_train, df_y_train)
    X, y = multi_month_2_one(df_X, df_y)

    # random forest benchmark
    print('Training a random forest regressor as the benchmark ...')
    rfr = RandomForestRegressor(max_depth=10, random_state=0, n_estimators=8000, n_jobs=-1)
    rfr.fit(X_train.values, y_train)
    y_pred = rfr.predict(X_test.values)
    rf_error = np.sqrt(np.sum((y_test - y_pred)**2)/len(y_pred))
    print('RMSLE of Random Forest Regressor is: ', rf_error)

    # hyper-parameter values are determined in "modeling.html"
    print('Training a XGBoost regressor with tuned hyper-parameters ...')
    model = XGBRegressor(learning_rate=0.01, subsample=0.9, colsample_bytree=0.7, max_depth=10, n_estimators=3718)
    model.fit(X_train, y_train, eval_metric="rmse")
    y_pred = model.predict(X_test)
    xgb_error = np.sqrt(np.sum((y_test - y_pred)**2)/len(y_pred))
    print('RMSLE of XGBoost Regressor is: ', xgb_error)

    print("Retrain the model on all the data in \"trainingDataset.csv\" and save model to \"xgb.dat\" ...")
    model.fit(X.values, y, eval_metric="rmse")
    pickle.dump(model, open("xgb.dat", "wb"))