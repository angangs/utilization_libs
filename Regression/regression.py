import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import matplotlib.pyplot as plt


def train_feature_selection_model(x_train, y_train, x_test, features):
    lgbmc = LGBMRegressor()
    lgbmc.fit(x_train, y_train)
    k_best = len(lgbmc.feature_importances_[lgbmc.feature_importances_ > np.percentile(lgbmc.feature_importances_, 75)])
    remaining_indices = lgbmc.feature_importances_.argsort()[-k_best:][::-1]
    x_train_selected = x_train.iloc[:, [i for i in remaining_indices]]
    x_test_selected = x_test.iloc[:, [i for i in remaining_indices]]
    feat_imp = pd.Series(lgbmc.feature_importances_, index=features)
    feat_imp.nlargest(k_best).plot(kind='barh', figsize=(8, 10))
    plt.show()
    return x_train_selected, x_test_selected


def train_lgbm(x_train, y_train, eval_set):
    lgbmc = LGBMRegressor(boosting_type='gbdt', n_estimators=3000)
    space = {
        'num_leaves': [int(i) for i in np.random.uniform(low=10.0, high=50.0, size=10)],
        'learning_rate': np.random.uniform(low=0.005, high=0.5, size=10),
        'min_child_samples': [int(i) for i in np.random.uniform(low=10, high=100, size=10)],
        'reg_alpha': np.random.uniform(low=0, high=1, size=5),
        'reg_lambda': np.random.uniform(low=0, high=1, size=5),
        'colsample_bytree': np.random.uniform(low=.1, high=.9, size=5),
    }
    # run randomized search
    n_iter_search = 50
    random_search = RandomizedSearchCV(lgbmc, param_distributions=space, n_iter=n_iter_search, cv=3, verbose=True)
    random_search.fit(x_train, y_train, eval_set=eval_set, eval_metric="mae", early_stopping_rounds=50, verbose=True)
    print("Best parameters for the model: {}".format(random_search.best_params_))

    """Fitting"""
    lgbmc = LGBMRegressor(**random_search.best_params_)
    lgbmc.fit(x_train, y_train, eval_set=eval_set, eval_metric="mae", early_stopping_rounds=50)
    return lgbmc


def regression_benchmarking(df_input, df_target, feature_selection=False):
    preds_buf = []
    err_buf = []

    """Split to training and test set"""
    x_train, x_test, y_train, y_test = train_test_split(df_input, df_target, test_size=.5, random_state=0)

    """Feature Selection"""
    if feature_selection:
        x_train, x_test = train_feature_selection_model(x_train, y_train, x_test, df_input.columns)

    """Training & Prediction"""
    regr = train_lgbm(x_train, y_train, eval_set=(x_test, y_test))
    prediction = regr.predict(x_test)
    preds_buf.append(prediction)

    """MAE"""
    print("MAE: " + str(mean_absolute_error(y_test, prediction)))
    err_buf.append(mean_absolute_error(y_test, prediction))

    """Dataframe prediction error"""
    df_pred_y_test = pd.DataFrame()
    df_pred_y_test['Prediction'] = prediction
    df_pred_y_test['Y_Test'] = y_test.values
    df_pred_y_test['Error'] = abs(df_pred_y_test['Prediction'] - df_pred_y_test['Y_Test'])
    print('Mean MAE = ' + str(np.mean(err_buf)) + ' +/- ' + str(np.std(err_buf)))

    return regr.predict(df_input[x_train.columns])
