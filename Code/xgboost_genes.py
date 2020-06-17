'''
Name:               Peter Hwang
Email:              hwangp@mit.edu
Project:            Machine Learning Cancer Diagnosis (eXtreme Gradient Boosting with Cross-Validation)
Date Completed:     June 15, 2020
'''

# Import Packages
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import recall_score, precision_score, plot_roc_curve

# Diagnosis can take the values 'PNI' or 'LVI', depending on which symptom is being examined
diagnosis = 'PNI'

# Load Datasets (PNI)
if diagnosis == 'PNI':
    train = pd.read_csv('train_PNI.csv')
    test = pd.read_csv('test_PNI.csv')
    features = list(train.columns.values)
    features.remove('PNI')
    target = 'PNI'
    IDcol = 'ID'

# Load Datasets (LVI)
elif diagnosis == 'LVI':
    train = pd.read_csv('train_LVI.csv')
    test = pd.read_csv('test_LVI.csv')
    features = list(train.columns.values)
    features.remove('LVI')
    target = 'LVI'
    IDcol = 'ID'

predictors = [x for x in train.columns if x not in [target, IDcol]]

def modelfit(alg, dtrain, dtest, predictors, useTrainCV = True, cv_folds = 5, early_stopping_rounds = 50):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(dtrain[predictors].values, label = dtrain[target].values)
        cvresult = xgb.cv(xgb_param, 
                          xgtrain, 
                          num_boost_round = alg.get_params()['n_estimators'], 
                          nfold = cv_folds,
                          metrics = 'auc', 
                          early_stopping_rounds = early_stopping_rounds,
                          verbose_eval = True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    # Fit the algorithm on the data
    alg.fit(dtrain[predictors], dtrain['PNI'], eval_metric = 'auc')
        
    # Predict training set:
    dtrain_predictions = alg.predict(dtrain[predictors])
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
    dtrain_df = pd.DataFrame(dtrain_predprob)
    np.savetxt("Probability_PNI_train.csv", dtrain_df, delimiter = ",", fmt = '%s', header = "PNI")

    # Predict testing set:
    dtest_predictions = alg.predict(dtest[predictors])
    dtest_predprob = alg.predict_proba(dtest[predictors])[:,1]
    dtest_df = pd.DataFrame(dtest_predprob)
    np.savetxt("Probability_PNI_test.csv", dtest_df, delimiter = ",", fmt = '%s', header = "PNI")
    
    # Create feature importances list
    imp = alg.feature_importances_
    imp_df = pd.DataFrame(imp)
    np.savetxt("Importances_PNI.csv", imp_df, delimiter = ",", fmt = '%s', header = "PNI")
    roc_plot(alg, dtest, dtest_predprob)
    X_train = dtrain
    X_test = dtest

    # Calculate precision and recall scores
    train_precision = precision_score(X_train[target].values, dtrain_predictions)
    print('Precision Score (Training): {:.6f}'.format(train_precision))
    test_precision = precision_score(X_test[target].values, dtest_predictions)
    print('Precision Score (Testing): {:.6f}'.format(test_precision))
    train_recall = recall_score(X_train[target].values, dtrain_predictions)
    print('Recall Score (Training): {:.6f}'.format(train_recall))
    test_recall = recall_score(X_test[target].values, dtest_predictions)
    print('Recall Score (Testing): {:.6f}'.format(test_recall))

# Create XGBoost model
def xgboost():
    xgb1 = XGBClassifier(
        learning_rate = 0.01,
        n_estimators = 5000,
        max_depth = 7,
        min_child_weight = 3,
        gamma = 0.2,
        subsample = 0.8,
        colsample_bytree = 0.9,
        objective = 'binary:logistic',
        nthread = 4,
        scale_pos_weight = 1,
        seed = 27)
    modelfit(xgb1, train, test, predictors)

# Plot Receiver Operating Characteristic curve
def roc_plot(alg, dtest, dtest_predprob):
    # Load training dataset
    df = pd.read_csv('train_PNI.csv')
    X = df.drop(['PNI'], axis = 1).values
    y = df['PNI'].values
    
    # Use StratifiedKFold cross-validation to plot curve
    cv = StratifiedKFold(n_splits = 5, random_state = 0, shuffle = True)
    classifier = XGBClassifier(
        learning_rate = 0.01,
        n_estimators = 1000,
        max_depth = 5,
        min_child_weight = 1,
        gamma = 0,
        subsample = 0.8,
        colsample_bytree = 0.8,
        objective = 'binary:logistic',
        nthread = 4,
        scale_pos_weight = 1,
        seed = 27)
    
    # Calculate ROC AUC metric value
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots()
    for i, (train, test) in enumerate(cv.split(X, y)):
        classifier.fit(X[train], y[train])
        viz = plot_roc_curve(classifier, X[test], y[test], alpha=0.3, lw=1, ax=ax)
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)
    
    # Plot ROC curve
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.8)
    mean_tpr = np.mean(tprs, axis = 0)
    mean_tpr[-1] = 1.0
    ax.plot(mean_fpr, mean_tpr, color = 'b', lw = 2, alpha = 0.8)
    std_tpr = np.std(tprs, axis = 0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color = 'grey', alpha = 0.2, label = r'$\pm$ 1 std. dev.')
    ax.set(xlim = [-.05, 1.05], ylim = [-.05, 1.05], title = "ROC Curve (PNI)")
    plt.savefig('ROC_PNI.png')
    plt.show()

# Use GridSearchCV to tune all parameters
def tune_all():
    parameters = {
            'max_depth': range(3, 10, 2),
            'min_child_weight': range(1, 6, 2),
            'gamma': [i / 10.0 for i in range(0, 5)],
            'subsample': [i / 10.0 for i in range(6, 10)],
            'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch1 = GridSearchCV(estimator = XGBClassifier(
            learning_rate = 0.1,
            n_estimators = 1000, 
            max_depth = 5,
            min_child_weight = 1,
            gamma = 0, 
            subsample = 0.8, 
            colsample_bytree = 0.8,
            objective = 'binary:logistic', 
            nthread = 4, 
            scale_pos_weight = 1, seed=27), 
            param_grid = parameters, 
            scoring = 'roc_auc',
            n_jobs = 4,
            iid = False, 
            cv = 5)
    gsearch1.fit(train[predictors],train[target])
    best_parameters = gsearch1.best_params_
    print(best_parameters)
    best_score = gsearch1.best_score_
    print(best_score)

# Use GridSearchCV to tune max_depth and min_child_weight
def tune_depth_weight():
    param_test1 = {
            'max_depth': range(3, 10, 2),
            'min_child_weight': range(1, 6, 2)
    }
    gsearch1 = GridSearchCV(estimator = XGBClassifier(
            learning_rate = 0.1, 
            n_estimators = 1000,
            max_depth = 5,
            min_child_weight = 1,
            gamma = 0, 
            subsample = 0.8, 
            colsample_bytree = 0.8,
            objective = 'binary:logistic', 
            nthread = 4, 
            scale_pos_weight = 1, seed=27), 
            param_grid = param_test1, 
            scoring = 'roc_auc',
            n_jobs = 4,
            iid = False, 
            cv = 5)
    gsearch1.fit(train[predictors],train[target])
    best_parameters = gsearch1.best_params_
    print(best_parameters)
    best_score = gsearch1.best_score_
    print(best_score)

# Use GridSearchCV to tune gamma
def tune_gamma():
    param_test2 = {
            'gamma': [i / 10.0 for i in range(0, 5)]
    }
    gsearch2 = GridSearchCV(estimator = XGBClassifier( 
            learning_rate = 0.1,
            n_estimators = 1000,
            max_depth = 5,
            min_child_weight = 1,
            gamma = 0,
            subsample = 0.8, 
            colsample_bytree = 0.8,
            objective = 'binary:logistic', 
            nthread = 4, 
            scale_pos_weight = 1, seed=27), 
            param_grid = param_test2, 
            scoring = 'roc_auc',
            n_jobs = 4,
            iid = False, 
            cv = 5)
    gsearch2.fit(train[predictors],train[target])
    best_parameters = gsearch2.best_params_
    print(best_parameters)
    best_score = gsearch2.best_score_
    print(best_score)

# Use GridSearchCV to tune subsample and colsample_bytree
def tune_subsample_colsample():
    param_test3 = {
     'subsample': [i / 10.0 for i in range(6, 10)],
     'colsample_bytree': [i / 10.0 for i in range(6, 10)]
    }
    gsearch3 = GridSearchCV(estimator = XGBClassifier( 
            learning_rate = 0.1,
            n_estimators = 1000,
            max_depth = 5,
            min_child_weight = 1,
            gamma = 0,
            subsample = 0.8,
            colsample_bytree = 0.8,
            objective = 'binary:logistic', 
            nthread = 4, 
            scale_pos_weight = 1, seed=27), 
            param_grid = param_test3, 
            scoring = 'roc_auc',
            n_jobs = 4,
            iid = False, 
            cv = 5)
    gsearch3.fit(train[predictors],train[target])
    best_parameters = gsearch3.best_params_
    print(best_parameters)
    best_score = gsearch3.best_score_
    print(best_score)

if __name__ == '__main__':
    xgboost()
    #tune_all()
    #tune_depth_weight()
    #tune_gamma()
    #tune_subsample_colsample()