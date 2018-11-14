import numpy as np
import pandas as pd
from sklearn import ensemble as es
from sklearn.model_selection import cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn import preprocessing

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

df = pd.read_excel('Telco-Churn.xlsx')
df = df[['tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'TotalDayMinutes', 'TotalEveMinutes', 'TotalNightMinutes', 'CustomerServiceCalls', 'TotalRevenue', 'Churn']]
df.columns = ['tenure', 'has_phone', 'MultipleLines', 'has_internet', 'usage_1', 'usage_2', 'usage_3', 'service_call', 'revenue', 'churn']
df['revenue'] = ((pd.to_numeric(df.revenue, errors='coerce').fillna(0))/df.tenure).fillna(0)
df['churn'] = 1.0*(df.churn=='Yes')

df1 = pd.read_csv('TRAIN.csv')
df1 = df1[['network_age','Total Spend in Months 1 and 2 of 2017', 'Total SMS Spend', 'Total Data Consumption', 'Total Unique Calls', 'Total Call centre complaint calls', 'Churn Status']]
df1.columns = ['tenure', 'revenue', 'usage_3', 'usage_2', 'usage_1', 'service_call', 'churn']
df1['revenue'] = pd.to_numeric(df1.revenue, errors='coerce').fillna(0)/2

df2 = pd.read_excel('Churn.xls')
df2 = df2[['Account Length', 'Day Mins', 'Eve Mins', 'Night Mins', 'Intl Mins', 'CustServ Calls', 'Churn']]
df2.columns = ['tenure', 'usage_1', 'usage_2', 'usage_3', 'usage_4', 'service_call', 'churn']

num_feature_list = ['tenure','revenue','service_call','Day_usage', 'Eve_usage', 'Night_usage', 'Intl_usage','sms', 'data_spend', 'data_consume', 'calls', 'onnet', 'offnet']
flag_feature_list = ['has_phone', 'MultipleLines', 'has_internet']
target_var = 'churn'

df[flag_feature_list] = df[flag_feature_list].apply(lambda x:1.0*(x!='No'))
df[intersection(df.columns, num_feature_list)] = preprocessing.scale(df[intersection(df.columns, num_feature_list)])
df1[intersection(df1.columns, num_feature_list)] = preprocessing.scale(df1[intersection(df1.columns, num_feature_list)])
df2[intersection(df2.columns, num_feature_list)] = preprocessing.scale(df2[intersection(df2.columns, num_feature_list)])

clf_rf = es.RandomForestClassifier(n_estimators=500, max_depth=3, max_features=6, class_weight='balanced_subsample')
clf_adb = es.AdaBoostClassifier()
clf_xb = es.GradientBoostingClassifier(n_estimators=100, subsample=0.9, max_features=6)
clf_svc = SVC(kernel='rbf', class_weight='balanced', probability=True)
clf_knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
clf_nn = MLPClassifier(hidden_layer_sizes=(64, ), alpha=1)

model = [clf_rf, clf_adb, clf_xb, clf_svc, clf_knn, clf_nn]
for data in [df, df1, df2]:
    print(data.columns)
    for m in model:
        scores = cross_validate(m, data.drop(columns=[target_var]), data[target_var], scoring=('roc_auc', 'neg_log_loss'), cv=10)
        print(m.__class__.__name__+' / roc_auc =', np.mean(scores['test_roc_auc']))
        print(m.__class__.__name__+' / log_loss =', np.mean(scores['test_neg_log_loss']))

clf_xb.fit(df.drop(columns=[target_var]), df[target_var])
for el in sorted(zip(clf_xb.feature_importances_,df.drop(columns=[target_var]).columns), reverse=True):
    print(el[0], '\t', el[1])

joblib.dump(clf_xb, 'model1.pkl' , compress=3)

clf_xb = es.GradientBoostingClassifier(n_estimators=100, subsample=0.9, max_features=6)
clf_xb.fit(df1.drop(columns=[target_var]), df1[target_var])
for el in sorted(zip(clf_xb.feature_importances_,df1.drop(columns=[target_var]).columns), reverse=True):
    print(el[0], '\t', el[1])

joblib.dump(clf_xb, 'model2.pkl' , compress=3)

clf_xb = es.GradientBoostingClassifier(n_estimators=100, subsample=0.9, max_features=6)
clf_xb.fit(df2.drop(columns=[target_var]), df2[target_var])
for el in sorted(zip(clf_xb.feature_importances_,df2.drop(columns=[target_var]).columns), reverse=True):
    print(el[0], '\t', el[1])

joblib.dump(clf_xb, 'model3.pkl' , compress=3)
