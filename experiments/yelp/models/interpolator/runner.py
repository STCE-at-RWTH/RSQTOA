import time

import pandas as pd

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from experiments.commons.regressors import InterpolatorRegressor

random_state = 123
validation_fraction = 0.05

config_path = 'config.yml'

feature_variables = ['total_reviews', 'elapsed_days_restaurant', 'ratio_positive_reviews', 'negative_reviews',
                     'immediacy_index']
target_variable = 'helpfulness_score_cbrt'

data = pd.read_csv('./../../data/data.csv')

kf = KFold(n_splits=20, shuffle=True, random_state=random_state)

df_results = pd.DataFrame(
    columns=['fold', 'train_time', 'train_mae', 'train_mse', 'train_rmse', 'train_r2', 'test_mae', 'test_mse',
             'test_rmse', 'test_r2'])

lower_limit = [-0.00001 for _ in feature_variables]
upper_limit = [1.00001 for _ in feature_variables]

fold = 1
for train_index, test_index in kf.split(range(data.shape[0])):
    print('Fold ::', fold)

    train_data, test_data = data.iloc[train_index], data.iloc[test_index]

    scaler = MinMaxScaler()
    train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data), columns=data.columns)

    features = train_data[feature_variables].astype('float32').to_numpy()
    target = train_data[target_variable].astype('float32').to_numpy()

    regressor = InterpolatorRegressor(config_path=config_path, lower=lower_limit, upper=upper_limit,
                                      validation_fraction=validation_fraction, random_state=random_state)

    start = time.time()
    regressor.fit(features=features, target=target)
    end = time.time()
    training_time = end - start

    print('Training Time :: ', training_time)

    regressor.plot_loss_history()

    train_pred = regressor.predict(features=features)

    train_mse = mean_squared_error(target, train_pred)
    train_rmse = train_mse ** .5
    train_r2 = r2_score(target, train_pred)
    train_mae = mean_absolute_error(target, train_pred)

    print('MA Error (Train) ::', train_mae)
    print('MS Error (Train) ::', train_mse)
    print('RMS Error (Train) ::', train_rmse)
    print('R^2 Score (Train) ::', train_r2)

    test_x = test_data[feature_variables].to_numpy()
    test_y = test_data[target_variable].to_numpy().astype('float32')

    test_pred = regressor.predict(test_x)

    test_mse = mean_squared_error(test_y, test_pred)
    test_rmse = test_mse ** .5
    test_r2 = r2_score(test_y, test_pred)
    test_mae = mean_absolute_error(test_y, test_pred)

    print('MA Error (Test) ::', test_mae)
    print('MS Error (Test) ::', test_mse)
    print('RMS Error (Test) ::', test_rmse)
    print('R^2 Score (Test) ::', test_r2)

    df_results = pd.concat([df_results, pd.DataFrame(
        [[fold, training_time, train_mae, train_mse, train_rmse, train_r2, test_mae, test_mse, test_rmse, test_r2]],
        columns=df_results.columns)], ignore_index=True)

    pd.DataFrame({
        'train_loss': regressor.loss_hist()['train_loss_points'][1],
        'val_loss': regressor.loss_hist()['val_test_loss_points'][1]
    }).to_csv('loss_hist_{}.csv'.format(fold))

    fold += 1

df_results.to_csv('results.csv')
