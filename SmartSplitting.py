import pyreadstat
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from scipy.spatial import distance, distance_matrix
from sklearn.impute import SimpleImputer
import pandas as pd
from skbio import DistanceMatrix
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.utils import class_weight
from xgboost import XGBClassifier
from imblearn import pipeline as pl

df, meta = pyreadstat.read_sav("CMS.sav")
cols = df.columns
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
df = df.select_dtypes(include=['float32', 'float64', 'int'])
imp.fit(df.values)
df = imp.transform(df.values)

CMS = np.round(df[:, -1])
# df = df.select_dtypes(include=['float32', 'float64', 'int'])
X = df[:, 0:df.shape[1] - 1:1]
CMS = CMS - 1
CMS[CMS == -1] = 1

Xd = pd.DataFrame(X)
duplicatedItem = Xd.duplicated(keep='first')
X = X[duplicatedItem == False, :]
CMS = CMS[duplicatedItem == False]

class1Data = X[CMS == 0, :]
class2Data = X[CMS == 1, :]
class1Target = CMS[CMS == 0]
class2Target = CMS[CMS == 1]

minorClassSize = class2Data.shape[0]
pipelines = []
for i in range(int(class1Data.shape[0] / minorClassSize)):
    X = np.append(class2Data, class1Data[range(i * minorClassSize, (i + 1) * minorClassSize), :], axis=0)
    CMS = np.append(class2Target, class1Target[range(i * minorClassSize, (i + 1) * minorClassSize)], axis=0)

    if i == int(class1Data.shape[0] / minorClassSize) - 1:  # test
        y_pred_test_all = np.zeros([CMS.shape[0], len(pipelines)])
        for i, pipelineItem in enumerate(pipelines):
            y_pred_test_all[:, i] = pipelineItem.predict(X)

        y_pred_test = np.min(y_pred_test_all, axis=1)
        acc = accuracy_score(y_pred_test, CMS)
        rec = recall_score(y_pred_test, CMS)
        f1Score = f1_score(y_pred_test, CMS)
        aucValue = roc_auc_score(y_pred_test, CMS)
        # accuracy.append(acc)
        # recall.append(rec)
        # fscore.append(f1Score)
        # auc.append(aucValue)

        print("Test Acc: {}".format(acc))
        print("Test recal: {}".format(rec))
        print("Test f1Score:{}".format(f1Score))
        print("Test AUC : {}".format(aucValue))
        continue

    smt = SMOTETomek()
    rus = RandomUnderSampler()
    ros = RandomOverSampler()

    pipeline = pl.make_pipeline(XGBClassifier())
    class_weights = class_weight.compute_class_weight('balanced',
                                                      np.unique(CMS),
                                                      CMS)
    accuracy = []
    recall = []
    fscore = []
    auc = []

    # sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
    # for train_index, test_index in sss.split(X, CMS):
    X_train, X_test, y_train, y_test = train_test_split(X, CMS, test_size=0.1, stratify=CMS)
    # df_ = resample(X_all, n_samples=500, replace=False, stratify=y_train)
    # y_ = np.round(df_[:, -1])
    # df = df.select_dtypes(include=['float32', 'float64', 'int'])
    # X_ = df_[:, 0:df_.shape[1] - 1:1]
    X_, y_ = ros.fit_sample(X_train, y_train)
    X_, y_ = rus.fit_sample(X_, y_)
    # X_, y_ = smt.fit_resample(X_, y_)
    # X_, y_ = resample(X_train, y_train,stratify=y_train,n_samples=1000)
    weights = np.zeros([1, len(y_)])
    weights[0, y_ == 0] = class_weights[0]
    weights[0, y_ == 1] = class_weights[1]
    pipeline.fit(X_, y_)
    pipelines.append(pipeline)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_pred, y_test)
    rec = recall_score(y_pred, y_test)
    f1Score = f1_score(y_pred, y_test)
    aucValue = roc_auc_score(y_pred, y_test)
    # accuracy.append(acc)
    # recall.append(rec)
    # fscore.append(f1Score)
    # auc.append(aucValue)

    print("Acc: {}".format(acc))
    print("recal: {}".format(rec))
    print("f1Score:{}".format(f1Score))
    print("AUC : {}".format(aucValue))
# pipeline.fit(newX, newY)
# y_pred_bal = pipeline.predict(X_test)
#
# # Show the classification report
# print(classification_report_imbalanced(y_test, y_pred_bal))

# pickle.dump(model, open("CMS.h5", 'wb'))
# PlotModel(model, filename='Anxiety', plot_neuron_name=True, view=True).plot()
