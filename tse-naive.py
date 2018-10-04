from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import logging
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from telus.src.models.pipelines import churn_features_pipeline
from telus.src.models.pipelines import churn_kaggle_features_pipeline
from telus.src.models.pipelines import cross_sell_features_pipeline
from telus.src.models.pipelines import cross_sell_kaggle_features_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.externals import joblib
from sklearn import preprocessing
import telus.src.config.config as mdl_cfg
from telus.src.validation.model_measurements import *
from telus.src.broker.s3_broker import S3Broker
from sklearn.utils import resample
from sklearn import metrics
from scipy import stats

def load_data_by_key(bucket, key):
    return spark.read.option("header", "true").option("inferSchema", "true").csv("s3://" + bucket + '/' + key)

def boostrap_measure(test_df, model, B_sample):
    f1 = []
    acc = []
    for i in range(B_sample):
        test_B = resample(test_df)
        test_probas = model.predict(test_B)
        f1.append(metrics.f1_score(test_df['TARGET'], test_probas))
        acc.append(metrics.accuracy_score(test_df['TARGET'], test_probas))
    m_f1 = [np.mean(f1), np.var(f1)]
    m_acc = [np.mean(acc), np.var(acc)]
    return m_f1, m_acc

def fit_kaggle(df, kaggle_model):
    feature_list = ['tenure', 'has_wireless', 'mobile_subscribers', 'has_internet', 'wls_billing_90day', 'wln_billing_90day']
    features = mdl_cfg.CHURN_FEATURES[prod]['continuous']
    df_kaggle = df[[f for f in feature_list if f in features]]
    df_kaggle.loc[:,'MonthlyCharges']=(df_kaggle['wls_billing_90day']+df_kaggle['wln_billing_90day'])/3
    df_kaggle.loc[:,'tenure']=df_kaggle['tenure']/30
    for f in [f for f in feature_list if f not in features]:
        df_kaggle.loc[:,f]=1

    df_kaggle.loc[:,['tenure', 'MonthlyCharges']] = preprocessing.scale(df_kaggle[['tenure', 'MonthlyCharges']])
    df_kaggle = df_kaggle[['has_wireless', 'mobile_subscribers', 'tenure', 'MonthlyCharges', 'has_internet']]
    df_kaggle.columns = ['PhoneService', 'MultipleLines', 'tenure', 'MonthlyCharges', 'InternetService']
    pp = kaggle_model.predict_proba(df_kaggle)
    df.loc[:,'prof_kaggle']=pp[:,1]
    return 1

logging.getLogger("py4j").setLevel(logging.ERROR)

spark = SparkSession \
    .builder \
    .appName("cross_sell_baseline_models") \
    .getOrCreate()

spark.sparkContext.setLogLevel("FATAL")

parser = argparse.ArgumentParser()
parser.add_argument("-prod", type=str)
parser.add_argument("-type", type=str)
parser.add_argument("-model", type=str)
args = parser.parse_args()

prod = args.prod
model = args.model   # kaggle model
model_type = args.type    # Telus model
bucket = mdl_cfg.S3['bucket']
s3 = S3Broker(bucket)

# LOAD TRAIN/TEST/VALIDATION SETS INTO PANDAS DF
train_df = load_data_by_key(bucket=bucket, key=mdl_cfg.S3['training_data'] + '{}/{}/train.csv'.format(prod, model_type)).toPandas()
test_df = load_data_by_key(bucket=bucket, key=mdl_cfg.S3['training_data'] + '{}/{}/test.csv'.format(prod, model_type)).toPandas()

# load professor Kaggle and add feature to train_df/test_df
rf_kaggle = joblib.load(model)
fit_kaggle(train_df, rf_kaggle)
fit_kaggle(test_df, rf_kaggle)

# CREATE PIPELINE
rf_clf = RandomForestClassifier(class_weight='balanced_subsample', n_estimators=200, random_state=42)
if model_type == 'churn':
    pipe = Pipeline([churn_features_pipeline[prod], ("estimator", rf_clf)])
    pipe_kaggle = Pipeline([churn_kaggle_features_pipeline[prod], ("estimator", rf_clf)])
elif model_type == 'cross_sell':
    pipe = Pipeline([cross_sell_features_pipeline[prod], ("estimator", rf_clf)])
    pipe_kaggle = Pipeline([cross_sell_kaggle_features_pipeline[prod], ("estimator", rf_clf)])
else:
    print('cannot recognize the model type')

rf_param_dist = {
    'estimator__max_features': list(range(3,7)),
    'estimator__max_depth': list(range(3,7))
}

random_search = RandomizedSearchCV(pipe, scoring=technical_lift_scorer, param_distributions=rf_param_dist,
                                   n_iter=10, n_jobs=-1, cv=5, random_state=42)

random_search_kaggle = RandomizedSearchCV(pipe_kaggle, scoring=technical_lift_scorer, param_distributions=rf_param_dist,
                                   n_iter=10, n_jobs=-1, cv=5, random_state=42)

model = random_search.fit(train_df, train_df['TARGET']).best_estimator_
model_k = random_search_kaggle.fit(train_df, train_df['TARGET']).best_estimator_

B_sample = 1000
m_f1, m_acc = boostrap_measure(test_df, model, B_sample)
m_f1_k, m_acc_k = boostrap_measure(test_df, model_k, B_sample)
norm_stat = np.abs(m_f1_k[0]-m_f1[0])/np.sqrt((m_f1[1] + m_f1_k[1])/B_sample)

print(m_f1)
print(m_f1_k)
print('f1 improved by:', (m_f1_k[0]-m_f1[0])/m_f1[0])
print('p_value:', 1-stats.norm.cdf(norm_stat))

# PRINT FEATURE IMPORTANCE
features = mdl_cfg.CHURN_FEATURES[prod]['continuous'] + mdl_cfg.CHURN_FEATURES[prod]['categorical']['province']
features_k = mdl_cfg.CHURN_FEATURES[prod]['continuous'] + ['prof_kaggle'] + mdl_cfg.CHURN_FEATURES[prod]['categorical']['province']

print("FEATURE IMPORTANCES:")
for el in sorted(zip(model.named_steps['estimator'].feature_importances_,features), reverse=True):
    print(el[0], '\t', el[1])
print()

print("FEATURE IMPORTANCES (with prof. Kaggle):")
for el in sorted(zip(model_k.named_steps['estimator'].feature_importances_,features_k), reverse=True):
    print(el[0], '\t', el[1])
print()

# PRINT TECHNICAL LIFTS BY VOLUME BASELINE CONVERSION FROM TRAIN
test_probas = model.predict_proba(test_df)[:,1]
test_probas_k = model_k.predict_proba(test_df)[:,1]
print("TECHNICAL LIFTS BY VOLUME:")
print(compute_lifts_by_decile_volume(test_probas=test_probas, test_labels=test_df['TARGET'], baseline_avg=train_df['TARGET'].mean()))
print()
print("TECHNICAL LIFTS BY VOLUME (with prof. Kaggle):")
print(compute_lifts_by_decile_volume(test_probas=test_probas_k, test_labels=test_df['TARGET'], baseline_avg=train_df['TARGET'].mean()))
print()

# PRINT TECHNICAL LIFTS BY VOLUME ON BASELINE CONVERSION FROM TEST
print("TECHNICAL LIFTS BY VOLUME with Test Set Baseline:")
print(compute_lifts_by_decile_volume(test_probas=test_probas, test_labels=test_df['TARGET'], baseline_avg=test_df['TARGET'].mean()))
print()
print("TECHNICAL LIFTS BY VOLUME with Test Set Baseline (with prof. Kaggle):")
print(compute_lifts_by_decile_volume(test_probas=test_probas_k, test_labels=test_df['TARGET'], baseline_avg=test_df['TARGET'].mean()))
print()

# PRINT CONVERSIONS BY DECILES
print("CONVERSIONS BY DECILES:")
print(compute_conversion_by_deciles(test_probas=test_probas, test_labels=test_df['TARGET']))
print()
print("CONVERSIONS BY DECILES (with prof. Kaggle):")
print(compute_conversion_by_deciles(test_probas=test_probas_k, test_labels=test_df['TARGET']))
print()
