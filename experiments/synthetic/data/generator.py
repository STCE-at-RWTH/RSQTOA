import random
import numpy as np
import pandas as pd

pd.set_option("display.precision", 15)


def f(x):
    return np.add(np.exp(np.add(x[0], x[1])), x[2]) + 3


def generate_test_samples(n: int = 100):
    x = np.array([[random.random(), random.random(), random.random()] for _ in range(n)])
    df = pd.DataFrame(np.array([x[:, 0], x[:, 1], x[:, 2], f(x.T)]).T, columns=['x0', 'x1', 'x2', 'z'], index=None)
    df.to_csv('data.csv')


generate_test_samples(10000)
