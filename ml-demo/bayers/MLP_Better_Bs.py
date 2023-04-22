from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as pyplot
from warnings import simplefilter
import joblib
import time
import warnings
warnings.filterwarnings('ignore')
#显示所有的列
pd.set_option('display.max_columns', None)

#显示所有的行
pd.set_option('display.max_rows', None)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

# 定义要优化的超参数空间
def objective(trial):
    hidden_layer_sizes = tuple(
        trial.suggest_categorical('hidden_layer_sizes', [(10,), (50,), (50,50)])
    )
    activation = trial.suggest_categorical('activation', ['relu', 'logistic', 'tanh'])
    alpha = trial.suggest_loguniform('alpha', 1e-6, 1e1)
    learning_rate_init = trial.suggest_loguniform('learning_rate_init', 1e-2,1e1)
    max_iter = trial.suggest_categorical('max_iter', [100, 200])

    # 创建MLP分类器对象
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter
    )

    # 计算交叉验证分数
    score = cross_val_score(mlp, x_train, y_train, cv=3).mean()

    return score


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
    y_train = dftrain['target'].values
    
    # 创建Optuna优化对象
    study = optuna.create_study(direction='minimize')

    # 开始优化
    study.optimize(objective, n_trials=100,n_jobs=12)

    # 输出最优参数和得分
    print("Best parameters:", study.best_params)
    print("Best score:", study.best_value)



