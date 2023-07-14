import time

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

random_state = 123
validation_fraction = 0.05
test_fraction = 0.05

feature_variables = ['total_reviews', 'elapsed_days_restaurant', 'ratio_positive_reviews', 'negative_reviews',
                     'immediacy_index']
target_variable = 'helpfulness_score_cbrt'

data = pd.read_csv('./../../data/data.csv')

kf = KFold(n_splits=20, shuffle=True, random_state=random_state)

df_results = pd.DataFrame(
    columns=['fold', 'train_time', 'train_mae', 'train_mse', 'train_rmse', 'train_r2', 'test_mae', 'test_mse',
             'test_rmse', 'test_r2'])

fold = 1
for train_index, test_index in kf.split(range(data.shape[0])):
    print('Fold ::', fold)

    train_data, test_data = pd.DataFrame(data.iloc[train_index], columns=data.columns), pd.DataFrame(
        data.iloc[test_index], columns=data.columns)

    pipeline_fold = Pipeline([
        ('scaler', MinMaxScaler()),
        ('polynomialfeatures', PolynomialFeatures(degree=4, include_bias=False)),
        ('elasticnet', ElasticNet(alpha=0.0001, l1_ratio=0.1))
    ])

    start = time.time()
    pipeline_fold.fit(train_data[feature_variables], train_data[target_variable])
    end = time.time()

    training_time = end - start
    print('Training Time :: ', training_time)

    train_pred = pipeline_fold.predict(train_data[feature_variables])

    train_mse = mean_squared_error(train_data[target_variable], train_pred)
    train_rmse = train_mse ** .5
    train_r2 = r2_score(train_data[target_variable], train_pred)
    train_mae = mean_absolute_error(train_data[target_variable], train_pred)

    print('MS Error (Train) ::', train_mse)
    print('RMS Error (Train) ::', train_rmse)
    print('MAE Error (Train) ::', train_mae)
    print('R^2 Score (Train) ::', train_r2)

    test_pred = pipeline_fold.predict(test_data[feature_variables])

    test_mse = mean_squared_error(test_data[target_variable], test_pred)
    test_rmse = test_mse ** .5
    test_r2 = r2_score(test_data[target_variable], test_pred)
    test_mae = mean_absolute_error(test_data[target_variable], test_pred)

    print('MS Error (Test) ::', test_mse)
    print('RMS Error (Test) ::', test_rmse)
    print('MAE Error (Test) ::', test_mae)
    print('R^2 Score (Test) ::', test_r2)

    df_results = pd.concat([df_results, pd.DataFrame(
        [[fold, training_time, train_mae, train_mse, train_rmse, train_r2, test_mae, test_mse, test_rmse, test_r2]],
        columns=df_results.columns)], ignore_index=True)

    fold += 1

df_results.to_csv('results.csv')
