import pickle
import pandas as pd
import pyreadstat
from gmdhpy.gmdh import Classifier
from gmdhpy.plot_model import PlotModel
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier, BalancedRandomForestClassifier
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.utils import class_weight, resample
from xgboost import XGBClassifier
from imblearn import pipeline as pl
from DataAnalysis import DataAnalysisUtils
from imblearn import over_sampling as os

df, meta = pyreadstat.read_sav("CMS.sav")
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
df = df.select_dtypes(include=['float32', 'float64', 'int'])
imp.fit(df.values)
df = imp.transform(df.values)

CMS = np.round(df[:, -1])
# df = df.select_dtypes(include=['float32', 'float64', 'int'])
X = df[:, 0:df.shape[1] - 1:1]

Xd = pd.DataFrame(X)
duplicatedItem = Xd.duplicated(keep='first')
X = X[duplicatedItem == False, :]
CMS = CMS[duplicatedItem == False]

x_train = X[CMS == 0, :]

clf = OneClassSVM(gamma='auto').fit(x_train)

y_pred = clf.predict(X)
y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1
accuracy = accuracy_score(y_pred, CMS)
recall = recall_score(y_pred, CMS)
fscore = f1_score(y_pred, CMS)
auc = roc_auc_score(y_pred, CMS)

print(accuracy)
print(recall)
print(fscore)
print(auc)
