import pickle
import pandas as pd
import pyreadstat

from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.utils import class_weight, resample
from tensorflow._api.v1 import train
from tffm import TFFMClassifier
from imblearn import pipeline as pl
from xgboost import XGBClassifier

df, meta = pyreadstat.read_sav("CMS.sav")
cols = df.columns
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df = df.select_dtypes(include=['float32', 'float64', 'int'])
imp.fit(df.values)
df = imp.transform(df.values)

CMS = np.round(df[:, -1])
# df = df.select_dtypes(include=['float32', 'float64', 'int'])
X = df[:, 0:df.shape[1] - 1:1]

# kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
# dA = DataAnalysisUtils()
# dA.plotCorrelationHeatMap(df)
# dA.plotVariableImportance(X, emotional)
# dA.plotPairAllFeatureByHue(df, "emotional_distress")
Xd = pd.DataFrame(X)
duplicatedItem = Xd.duplicated(keep='first')
X = X[duplicatedItem == False, :]
CMS = CMS[duplicatedItem == False]

# X = np.append(X, np.reshape(np.sum(X, axis=1), (X.shape[0], 1)), axis=1)
X = X + range(10, X.shape[1] + 10)
X_train, X_test, y_train, y_test = train_test_split(X, CMS, test_size=0.1, stratify=CMS)
# transformer = PCA(n_components=30)
# X_train = transformer.fit_transform(X_train)
# X_test = transformer.fit_transform(X_test)

smt = SMOTETomek()
rus = RandomUnderSampler()
ros = RandomOverSampler()
# brf = BalancedRandomForestClassifier(n_estimators=50)
# base_estimator = AdaBoostClassifier(n_estimators=20)

pipeline = pl.make_pipeline(XGBClassifier())

class_weights = class_weight.compute_class_weight('balanced',
                                                  np.unique(CMS),
                                                  CMS)
accuracy = []
recall = []
fscore = []
auc = []
bootstrap_iter = 10
X_all = np.append(X_train, np.reshape(y_train, (X_train.shape[0], 1)), axis=1)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
for train_index, test_index in sss.split(X, CMS):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index, :], X[test_index, :]
    y_train, y_test = CMS[train_index], CMS[test_index]
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
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_pred, y_test)
    rec = recall_score(y_pred, y_test)
    f1Score = f1_score(y_pred, y_test)
    aucValue = roc_auc_score(y_pred, y_test)
    accuracy.append(acc)
    recall.append(rec)
    fscore.append(f1Score)
    auc.append(aucValue)

print(accuracy)
print(recall)
print(fscore)
print(auc)
# pipeline.fit(newX, newY)
# y_pred_bal = pipeline.predict(X_test)
#
# # Show the classification report
# print(classification_report_imbalanced(y_test, y_pred_bal))

# pickle.dump(model, open("CMS.h5", 'wb'))
# PlotModel(model, filename='Anxiety', plot_neuron_name=True, view=True).plot()
pass
