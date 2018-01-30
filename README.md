# [Online Sales Prediction](https://www.kaggle.com/c/online-sales) 

To generate a submission, first run "create_features.py",
then run "create_model.py", at last run "prediction.py".
Data exploration process is shown in "exploration.html".
Parameter tuning process is shown in "tuning.html"

## create_features.py:
Conduct feature engineering on "TrainingDataset.csv" and 
"TestDataset.csv". Generate two csv files: "clean_data_train.csv" and "clean_data_test.csv"

## create_model.py:
Read in "clean_data_train.csv", build a model by using the
hypter-parameters determined in "tuning.html".
Write the model to disk as "xgb.dat"

## prediction.py:
Read in "clean_data_test.csv", "sample_submission.csv" and
"xgb.dat". Create "submission.csv" for submission.   
