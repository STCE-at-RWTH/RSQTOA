import time

import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

random_state = 123
validation_fraction = 0.05
test_fraction = 0.05

feature_variables = ['x0', 'x1', 'x2']
target_variable = 'z'

data = pd.read_csv('../../data/data.csv')

kf = KFold(n_splits=20, shuffle=True, random_state=random_state)

df_results = pd.DataFrame(
    columns=['fold', 'train_time', 'train_mae', 'train_mse', 'train_rmse', 'train_r2', 'test_mae', 'test_mse',
             'test_rmse', 'test_r2'])

fold = 1
for train_index, test_index in kf.split(range(data.shape[0])):
    print('Fold ::', fold)

    train_data, test_data = data.iloc[train_index], data.iloc[test_index]

    scaler = MinMaxScaler()
    train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data), columns=data.columns)

    train_x = train_data[feature_variables].values
    train_y = train_data[target_variable].values

    regressor = LinearRegression(fit_intercept=True)

    start = time.time()
    regressor.fit(train_x, train_y)
    end = time.time()
    training_time = end - start

    print('Training Time :: ', training_time)

    train_pred = regressor.predict(train_x)

    train_mse = mean_squared_error(train_y, train_pred)
    train_rmse = train_mse ** .5
    train_r2 = r2_score(train_y, train_pred)
    train_mae = mean_absolute_error(train_y, train_pred)

    print('MS Error (Train) ::', train_mse)
    print('RMS Error (Train) ::', train_rmse)
    print('MAE Error (Train) ::', train_mae)
    print('R^2 Score (Train) ::', train_r2)

    test_x = test_data[feature_variables].to_numpy()
    test_y = test_data[target_variable].to_numpy().astype('float32')

    test_pred = regressor.predict(test_x)

    test_mse = mean_squared_error(test_y, test_pred)
    test_rmse = test_mse ** .5
    test_r2 = r2_score(test_y, test_pred)
    test_mae = mean_absolute_error(test_y, test_pred)

    print('MS Error (Test) ::', test_mse)
    print('RMS Error (Test) ::', test_rmse)
    print('MAE Error (Test) ::', test_mae)
    print('R^2 Score (Test) ::', test_r2)

    df_results = pd.concat([df_results, pd.DataFrame(
        [[fold, training_time, train_mae, train_mse, train_rmse, train_r2, test_mae, test_mse, test_rmse, test_r2]],
        columns=df_results.columns)], ignore_index=True)

    fold += 1

df_results.to_csv('results.csv')
