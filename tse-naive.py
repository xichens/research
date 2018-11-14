from pyspark.sql.types import *
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
import logging
import pandas as pd
import numpy as np
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from telus.src.models.pipelines import churn_features_pipeline
from telus.src.models.pipelines import cross_sell_features_pipeline
import pipe as pip
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
import tse_config as cfg

def load_data_by_key(bucket, key):
    return spark.read.option("header", "true").option("inferSchema", "true").csv("s3://" + bucket + '/' + key)

def boostrap_tech_lift(test_df, target, model, B_sample):
    lift = []
    for i in range(B_sample):
        test_B, target_B = resample(test_df, target)
        test_probas = model.predict_proba(test_B)[:,1]
        tech_lift = compute_lifts_by_decile_volume(test_probas=test_probas, test_labels=target_B, baseline_avg=train_df['TARGET'].mean())
        lift.append(tech_lift[0])
    lift_mean = np.mean(lift)
    lift_std = np.std(lift)
    lift_lower = lift_mean - 1.96*lift_std
    lift_upper = lift_mean + 1.96*lift_std
    return lift_mean, lift_std, [lift_lower, lift_upper]

def boostrap_measure(test_df, model, B_sample):
    f1 = []
    log_loss = []
    for i in range(B_sample):
        test_B = resample(test_df)
        test_probas = model.predict_proba(test_B)
        test_pred = model.predict(test_B)
        f1.append(metrics.f1_score(test_B['TARGET'], test_pred))
        log_loss.append(metrics.log_loss(test_B['TARGET'], test_probas))
    m_f1 = [np.mean(f1), np.var(f1)]
    m_log_loss = [np.mean(log_loss), np.var(log_loss)]
    return [m_f1, m_log_loss]

def fit_kaggle(df, kaggle_model, prod):
    # df is the dataframe, either train_df or test_df
    # kaggle_model is a list of pre-trained and pickled model
    data = {}
    for i in cfg.feature_process['fix']:
        data.update({i: df[cfg.feature_process['fix'][i]]})
    for i in cfg.feature_process['avg']:
        data.update({i: df[cfg.feature_process['avg'][i]].mean(axis=1)})
    for i in cfg.feature_process['flag']:
        data.update({i: 1.0*(df[cfg.feature_process['flag'][i]]>0)})
    df_kaggle = pd.DataFrame(data=data)
    df_kaggle[cfg.feature_process['norm']] = preprocessing.scale(df_kaggle[cfg.feature_process['norm']])
    log_score = []
    roc_auc_score = []
    for model in kaggle_model:
        prof_kaggle = joblib.load('telus/' + model + '.pkl')
        pp = prof_kaggle.predict_proba(df_kaggle[[cfg.prod_feature[prod][i] for i in cfg.feature[model]]])
        df.loc[:,model]=pp[:,1]
        log_score.append(metrics.log_loss(df['TARGET'], pp[:,1]))
        roc_auc_score.append(metrics.roc_auc_score(df['TARGET'], pp[:,1]))
    return [log_score, roc_auc_score]

logging.getLogger("py4j").setLevel(logging.ERROR)

spark = SparkSession \
    .builder \
    .appName("cross_sell_baseline_models") \
    .getOrCreate()

spark.sparkContext.setLogLevel("FATAL")

parser = argparse.ArgumentParser()
parser.add_argument("-prod", type=str)
parser.add_argument("-type", type=str)
args = parser.parse_args()

prod = args.prod
model_type = args.type    # Telus model
bucket = mdl_cfg.S3['bucket']
s3 = S3Broker(bucket)

print('working on {}-{} model'.format(prod, model_type))
print()

# LOAD TRAIN/TEST/VALIDATION SETS INTO PANDAS DF
train_df = load_data_by_key(bucket=bucket, key=mdl_cfg.S3['training_data'] + '{}/{}/train'.format(prod, model_type)).toPandas()
test_df = load_data_by_key(bucket=bucket, key=mdl_cfg.S3['training_data'] + '{}/{}/test'.format(prod, model_type)).toPandas()

# load professor Kaggle and add feature to train_df/test_df
kaggle_model = ['model1', 'model2', 'model3']
score = fit_kaggle(train_df, kaggle_model, prod)
score1 = fit_kaggle(test_df, kaggle_model, prod)
print('roc_auc score for train is {}'.format(score[1]))
print('roc_auc score for test is {}'.format(score1[1]))

# CREATE PIPELINE
rf_clf = RandomForestClassifier(class_weight='balanced_subsample', n_estimators=200, random_state=42)
if model_type == 'churn':
    pipe = Pipeline([churn_features_pipeline[prod], ("estimator", rf_clf)])
    pipe_kaggle = Pipeline([pip.churn_features_pipeline[prod], ("estimator", rf_clf)])
elif model_type == 'cross_sell':
    pipe = Pipeline([cross_sell_features_pipeline[prod], ("estimator", rf_clf)])
    pipe_kaggle = Pipeline([pip.cross_sell_features_pipeline[prod], ("estimator", rf_clf)])
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

# determine the stacking weights
telus_probas = model.predict_proba(train_df)[:,1]
weight_df = train_df[kaggle_model+['TARGET']]
weight_df.loc[:,'telus'] = telus_probas
weight_model = LogisticRegression(class_weight='balanced')
weight_model.fit(weight_df.drop(columns=['TARGET']), weight_df['TARGET'])

# boostrap technical_lift for the first decile
B_sample = 1000

# PRINT TECHNICAL LIFTS BY VOLUME BASELINE CONVERSION FROM TRAIN
test_probas = model.predict_proba(test_df)[:,1]
telus_lift_decile_1 = compute_lifts_by_decile_volume(test_probas=test_probas, test_labels=test_df['TARGET'], baseline_avg=train_df['TARGET'].mean())[0]
print('Telus lift benchmark: {}'.format(telus_lift_decile_1))
print()

stack_df = test_df[kaggle_model]
stack_df.loc[:,'telus'] = test_probas
lift_mean, lift_std, lift_interval = boostrap_tech_lift(test_df, test_df['TARGET'], model_k, B_sample)
print('TSE_additional_feature lift mean: {}'.format(lift_mean))
print('TSE_additional_feature lift interval: {}'.format(lift_interval))
print()
lift_mean, lift_std, lift_interval = boostrap_tech_lift(stack_df, test_df['TARGET'], weight_model, B_sample)
print('TSE_stacking lift mean: {}'.format(lift_mean))
print('TSE_stacking lift interval: {}'.format(lift_interval))
print()

# # PRINT FEATURE IMPORTANCE
# features = mdl_cfg.CHURN_FEATURES[prod]['continuous'] + mdl_cfg.CHURN_FEATURES[prod]['categorical']['province']
# features_k = mdl_cfg.CHURN_FEATURES[prod]['continuous'] + kaggle_model + mdl_cfg.CHURN_FEATURES[prod]['categorical']['province']
#
# print("FEATURE IMPORTANCES (with prof. Kaggle):")
# for el in sorted(zip(model_k.named_steps['estimator'].feature_importances_,features_k), reverse=True):
#     print(el[0], '\t', el[1])
# print()
#
# # PRINT TECHNICAL LIFTS BY VOLUME BASELINE CONVERSION FROM TRAIN
# test_probas = model.predict_proba(test_df)[:,1]
# test_probas_k = model_k.predict_proba(test_df)[:,1]
# test_probas_k_stack = 0.8*(model_k.predict_proba(test_df)[:,1]) + 0.2*(test_df[kaggle_model].multiply(weight).mean(axis=1))
# print("TECHNICAL LIFTS BY VOLUME:")
# print(compute_lifts_by_decile_volume(test_probas=test_probas, test_labels=test_df['TARGET'], baseline_avg=train_df['TARGET'].mean()))
# print()
# print("TECHNICAL LIFTS BY VOLUME (with prof. Kaggle):")
# print(compute_lifts_by_decile_volume(test_probas=test_probas_k, test_labels=test_df['TARGET'], baseline_avg=train_df['TARGET'].mean()))
# print()
#
# # PRINT TECHNICAL LIFTS BY VOLUME ON BASELINE CONVERSION FROM TEST
# print("TECHNICAL LIFTS BY VOLUME with Test Set Baseline:")
# print(compute_lifts_by_decile_volume(test_probas=test_probas, test_labels=test_df['TARGET'], baseline_avg=test_df['TARGET'].mean()))
# print()
# print("TECHNICAL LIFTS BY VOLUME with Test Set Baseline (with prof. Kaggle):")
# print(compute_lifts_by_decile_volume(test_probas=test_probas_k, test_labels=test_df['TARGET'], baseline_avg=test_df['TARGET'].mean()))
# print()
#
# # PRINT CONVERSIONS BY DECILES
# print("CONVERSIONS BY DECILES:")
# print(compute_conversion_by_deciles(test_probas=test_probas, test_labels=test_df['TARGET']))
# print()
# print("CONVERSIONS BY DECILES (with prof. Kaggle):")
# print(compute_conversion_by_deciles(test_probas=test_probas_k, test_labels=test_df['TARGET']))
# print()
# print("CONVERSIONS BY DECILES (with prof. Kaggle stacked):")
# print(compute_conversion_by_deciles(test_probas=test_probas_k_stack, test_labels=test_df['TARGET']))
# print()
