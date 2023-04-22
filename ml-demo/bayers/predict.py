from warnings import simplefilter
import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as pyplot
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
import time

dftest = pd.read_csv("./test30_reduced.csv")
simplefilter(action='ignore', category=FutureWarning)
seed = 7

class_names = dftest.target.unique()
print(class_names)

dftest=dftest.astype('category')
cat_columns = dftest.select_dtypes(['category']).columns
dftest[cat_columns] = dftest[cat_columns].apply(lambda x: x.cat.codes)


x_columns = dftest.columns.drop('target')
x_test = dftest[x_columns].values
y_test = dftest['target']

model_map = {
             "DecisionTreeClassifier": "./models/decision-tree.clf",
             "RandomForestClassifier": "./models/random-forest.clf",
             "MLPClassifier": "./models/mlp.clf",
             "GradientBoostingClassifier": "./models/gbdt.clf",
             "GaussianNB": "./models/gnb.clf"
             }


if __name__ == '__main__':

    #选定模型并加载训练好的模型
    model_name = "MLPClassifier"
    model = joblib.load(model_map.get(model_name))
    starttest = time.time()
    y_pred_dt = model.predict(x_test)
    # y_pred_dt_roc = model.predict_proba(x_test)
    endtest =time.time()
    difftest = endtest-starttest
    print("Test time: " + str(difftest))

    #计算准确率和f1
    print("{}, accuracy: ".format(model_name) + str(metrics.accuracy_score(y_test, y_pred_dt)) + " F1 score:" + str(metrics.f1_score(y_test, y_pred_dt,average='weighted')))
    #计算混淆矩阵并画出
    matrixdt = confusion_matrix(y_test,y_pred_dt)
    print(matrixdt)
    plot_confusion_matrix(model,x_test,y_test)
    pyplot.show()