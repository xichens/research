import numpy as np
import pandas as pd
from sklearn import ensemble as es
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import preprocessing

df = pd.read_csv('telco_2.csv')

num_feature_list = ['tenure','MonthlyCharges','InternetService']
cat_feature_list = ['PhoneService','MultipleLines']
target_var = 'Churn'

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce').fillna(0)
df['InternetService'] = 1.0*(df['InternetService']!='No')
df[['tenure','MonthlyCharges']] = preprocessing.scale(df[['tenure','MonthlyCharges']])

target = 1.0*(df[target_var]=='Yes')
features = pd.concat([pd.get_dummies(df[cat_feature_list], drop_first=True), df[num_feature_list]], axis=1).drop(columns=['MultipleLines_No phone service'])
features.columns = ['PhoneService', 'MultipleLines', 'tenure', 'MonthlyCharges', 'InternetService']

clf_rf = es.RandomForestClassifier(n_estimators=500, max_depth=3, max_features=5, class_weight='balanced_subsample')
clf_adb = es.AdaBoostClassifier()
clf_xb = es.GradientBoostingClassifier(n_estimators=100, subsample=0.9, max_features=5)
clf_mlp_1 = MLPClassifier(hidden_layer_sizes=(16, ), max_iter=500)
clf_mlp_2 = MLPClassifier(hidden_layer_sizes=(4, 12, ), max_iter=200)
clf_svc = SVC(kernel='rbf', class_weight='balanced', probability=True)

model = [clf_rf, clf_adb, clf_xb, clf_mlp_1, clf_mlp_2, clf_svc]
scores = []
for i in range(len(model)):
    scores.append(cross_validate(model[i], features, target, scoring=('f1', 'accuracy'), cv=10))
    print(model[i].__class__.__name__+' / F1_score =', np.mean(scores[i]['test_f1']))
    print(model[i].__class__.__name__+' / accuracy =', np.mean(scores[i]['test_accuracy']))

clf_rf.fit(features, target)
clf_svc.fit(features, target)

for el in sorted(zip(clf_rf.feature_importances_,features.columns), reverse=True):
    print(el[0], '\t', el[1])

joblib.dump(clf_rf, 'rf_kaggle.pkl' , compress=3)
joblib.dump(clf_svc, 'svc_kaggle.pkl' , compress=3)
