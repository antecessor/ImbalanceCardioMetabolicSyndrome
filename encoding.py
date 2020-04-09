import pickle
import pandas as pd
import pyreadstat
from gmdhpy.gmdh import Classifier
from gmdhpy.plot_model import PlotModel
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import BalancedBaggingClassifier, RUSBoostClassifier, BalancedRandomForestClassifier
from imblearn.metrics import classification_report_imbalanced
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks, RandomUnderSampler
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FastICA, PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight, resample
from xgboost import XGBClassifier
from imblearn import pipeline as pl
from DataAnalysis import DataAnalysisUtils
from imblearn import over_sampling as os
from category_encoders import *

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

transformer = PCA(n_components=30)
X = transformer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, CMS, test_size=0.3, stratify=CMS)

# create a pipeline
ppl = Pipeline([
    ('enc', JamesSteinEncoder(return_df=False, verbose=True, sigma=0.5)),
    ('norm', StandardScaler()),
    ("over", SMOTETomek()),
    ('clf', XGBClassifier())
])

# set the parameters by cross-validation
tuned_parameters = {
    'enc__sigma': np.linspace(0.05, 1, 10)
}

scores = ['recall']

for score in scores:
    print("# Tuning hyper-parameters for %s\n" % score)
    clf = GridSearchCV(ppl, tuned_parameters, cv=3, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:\n")
    print(clf.best_params_)
    print("\nGrid scores on development set:\n")
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%s (+/-%s) for %s" % (mean, std * 2, params))

    print("\nDetailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n")
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
