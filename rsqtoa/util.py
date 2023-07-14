import random
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

pd.set_option("display.precision", 15)


def f(x):
    tmp = 2 + x[0] * np.sin(5 * x[0])
    if np.isscalar(x[1]):
        if x[1] > 0.3:
            tmp += 10 * (x[1] - 0.3) ** 2
    else:
        tmp[x[1] > 0.3] += 10 * (x[1][x[1] > 0.3] - 0.3) ** 2

    return tmp


def generate_random_samples(n: int = 100, file_name: str = 'test_sample_{}.csv'):
    x = np.array([[random.random(), random.random()] for _ in range(n)])
    df = pd.DataFrame(np.array([x[:, 0], x[:, 1], f(x.T)]).T, columns=['x', 'y', 'z'], index=None)
    df.to_csv(file_name.format(n))


def generate_train_samples(n: int = 1000):
    X = np.arange(0, 1, 1 / n)
    Y = np.arange(0, 1, 1 / n)
    X, Y = np.meshgrid(X, Y)
    Z = f(np.stack((X.flatten(), Y.flatten())))
    df = pd.DataFrame(np.array([X.flatten(), Y.flatten(), Z]).T, columns=['x', 'y', 'z'], index=None)
    df.to_csv('train_sample_{}.csv'.format(n**2))


# for continuous data
def get_stratified_sample(df, target_col='z', no_of_items=10000):
    df['Fold'] = -1
    bins = 10000
    splits = int(df.shape[0] / no_of_items)

    skf = StratifiedKFold(n_splits=splits, shuffle=True)
    df['grp'] = pd.cut(df[target_col], bins, labels=False)
    target = df.grp

    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        df.loc[v, 'Fold'] = fold_no
    return df[df['Fold'] == 0].iloc[:, 1:4]
