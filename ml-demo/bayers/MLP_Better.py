from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from matplotlib import pyplot as pyplot
from warnings import simplefilter
import joblib
import time

def hyperparameter_tune(clf, parameters, iterations, X, y):
    randomSearch = RandomizedSearchCV(clf, param_distributions=parameters, n_jobs=-1, n_iter=iterations, cv=2)
    randomSearch.fit(X,y)
    params = randomSearch.best_params_
    score = randomSearch.best_score_
    return params, score


if __name__ == '__main__':
    # 加载训练数据和测试数据
    dftrain = pd.read_csv("./train70_reduced.csv")

    simplefilter(action='ignore', category=FutureWarning)
    seed = 7

    # 获取和打印数据总的类别数
    class_names = dftrain.target.unique()
    print(class_names)

    # 转换为分类数据,即将数据中标签形式数据转换为可编码类型
    dftrain = dftrain.astype('category')  # 可以指定特定的列转为分类数据 df['col1'] = df['col1'].astype('category')

    # 找出和打印分类标签列
    cat_columns = dftrain.select_dtypes(['category']).columns
    print('cat_columns ------------', cat_columns)

    # 将标签列转换为编码数字格式，以方便输入模型
    dftrain[cat_columns] = dftrain[cat_columns].apply(lambda x: x.cat.codes)

    # 分离目标y和特征列x
    x_columns = dftrain.columns.drop('target')
    x_train = dftrain[x_columns].values
    y_train = dftrain['target']

    # Multi layer perceptron
    print("Starting Multi layer perceptron")

    parameters = {
          'solver': ['sgd', 'adam', 'lbfgs'],
          'activation': ['relu', 'identity', 'logistic', 'tanh'],
          'learning_rate': ['constant', 'invscaling', 'adaptive']
      }

    clf = MLPClassifier(batch_size=256, verbose=True, early_stopping=True)
    parameters_after_tuning, score_after_tuning = hyperparameter_tune(clf, parameters, 20, x_train, y_train);
    print(parameters_after_tuning)
    print(score_after_tuning)



