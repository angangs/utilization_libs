import pickle
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


output_folder = 'ml_results'
Path(output_folder).mkdir(parents=True, exist_ok=True)
now = datetime.now()
dt = now.strftime("%Y_%m_%d-%H_%M_%S")


def train_feature_selection_model(x_train, y_train, x_test, features, way='importance', k_best=10):
    if way == 'univariate':
        fs = SelectKBest(score_func=f_regression, k=k_best)
        fs.fit(x_train, y_train)
        x_train_selected = fs.transform(x_train)
        x_test_selected = fs.transform(x_test)
        feat_imp = pd.Series(fs.scores_, index=features)
        feat_imp.nlargest(k_best).plot(kind='barh', figsize=(8, 10))
        plt.show()
    elif way == 'hand':
        x_train_selected = x_train[:, :k_best]
        x_test_selected = x_test[:, :k_best]
    elif way == 'importance':
        lgbmc = LGBMClassifier()
        lgbmc.fit(x_train, y_train)
        k_best = len(lgbmc.feature_importances_[lgbmc.feature_importances_ >
                                                np.percentile(lgbmc.feature_importances_, 75)])
        remaining_indices = lgbmc.feature_importances_.argsort()[-k_best:][::-1]
        x_train_selected = x_train[:, [i for i in remaining_indices]]
        x_test_selected = x_test[:, [i for i in remaining_indices]]
        feat_imp = pd.Series(lgbmc.feature_importances_, index=features)
        feat_imp.to_csv('ml_results/feature_importance_bin_{}.csv'.format(dt), sep=';', encoding='utf-8-sig',
                        index=True)
        feat_imp.nlargest(k_best).plot(kind='barh', figsize=(8, 10))
        plt.show()
    else:
        x_train_selected = x_train
        x_test_selected = x_test

    return x_train_selected, x_test_selected


def train_lgbm(x_train, y_train, eval_set):
    lgbmc = LGBMClassifier(boosting_type='gbdt', n_estimators=3000)
    space = {
        'num_leaves': [int(i) for i in np.random.uniform(low=10.0, high=50.0, size=10)],
        'learning_rate': np.random.uniform(low=0.005, high=0.5, size=10),
        'min_child_samples': [int(i) for i in np.random.uniform(low=10, high=100, size=10)],
        'reg_alpha': np.random.uniform(low=0, high=1, size=5),
        'reg_lambda': np.random.uniform(low=0, high=1, size=5),
        'colsample_bytree': np.random.uniform(low=.1, high=.9, size=5),
    }
    # run randomized search
    n_iter_search = 30
    random_search = RandomizedSearchCV(lgbmc, param_distributions=space, n_iter=n_iter_search, cv=3)
    random_search.fit(x_train, y_train, eval_set=eval_set, eval_metric="logloss", early_stopping_rounds=50)
    print("Best parameters for the model: {}".format(random_search.best_params_))
    """Fitting"""
    lgbmc = LGBMClassifier(**random_search.best_params_)
    lgbmc.fit(x_train, y_train, eval_set=eval_set, eval_metric="logloss", early_stopping_rounds=50)
    return lgbmc


def get_metrics(y_test, prediction):
    y_test = y_test.astype(int)
    prediction = prediction.astype(int)
    auc_score = roc_auc_score(y_test, prediction, average=None)
    prec_score = precision_score(y_test, prediction, average=None)
    rec_score = recall_score(y_test, prediction, average=None)
    print("AUC: " + str(auc_score))
    print("Precision: " + str(prec_score))
    print("Recall: " + str(rec_score))
    return auc_score, prec_score, rec_score


def balance_data(x, y, shuffle_enabled=True, sampling='under'):
    if sampling == 'over':

        sampler = SMOTE(k_neighbors=5)
    else:
        sampler = RandomUnderSampler(random_state=0)

    x_resampled, y_resampled = sampler.fit_resample(x, y)
    data_resampled = np.column_stack((x_resampled, y_resampled))

    if shuffle_enabled:
        np.random.shuffle(data_resampled)
        x_resampled = data_resampled[:, :-1]
        y_resampled = data_resampled[:, -1]
    return x_resampled, y_resampled


def main():
    """Read Dataset"""
    score_table = pd.read_csv('ml_dataset/BranchPerfScoreExtProcessedOpapClf90.csv', sep=';', index_col=None,
                              encoding='utf-8-sig')

    """Define Input and Target data"""
    x = np.array(score_table.drop(['Label', 'Municipality', 'Score', 'Legacy GGR'], axis=1))
    y = np.array(score_table[['Score', 'Legacy GGR']])

    """Build Model"""
    n_iters = 5
    auc_score_list = []
    prec_score_list = []
    rec_score_list = []

    for i in range(n_iters):
        """Split to training and test set"""
        print("Split to training and test set")
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.10, random_state=i)
        x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=.2, random_state=i)
        y_train = y_train[:, 0]
        y_valid = y_valid[:, 0]

        """Balance Data"""
        # x_train, y_train = balance_data(x_train, y_train, shuffle_enabled=True, sampling='over')

        """Feature Selection"""
        print("Feature Selection")
        x_train, x_test = train_feature_selection_model(
            x_train, y_train, x_test, score_table.drop(['Label', 'Municipality', 'Score', 'Legacy GGR'],
                                                       axis=1).columns.tolist(), way='importance', k_best=10)

        """Training & Prediction"""
        print("Training & Prediction")
        clf = train_lgbm(x_train, y_train, [(x_train, y_train), (x_valid, y_valid)])
        prediction = clf.predict(x_test)
        # save the model to disk
        filename = 'ml_results/binaryclass_model_{}.pkl'.format(dt)
        pickle.dump(clf, open(filename, 'wb'))

        """Metric"""
        df = pd.DataFrame(prediction, columns=['Prediction'])
        df['Actual'] = y_test[:, 0]
        auc_score, prec_score, rec_score = get_metrics(df['Actual'], df['Prediction'])
        auc_score_list.append(auc_score)
        prec_score_list.append(prec_score)
        rec_score_list.append(rec_score)

    auc_score = np.mean(auc_score_list)
    prec_score = np.mean(prec_score_list)
    rec_score = np.mean(rec_score_list)

    print('Mean AUC:'+str(auc_score))
    print('Mean Precision:' + str(prec_score))
    print('Mean Recall:' + str(rec_score))

    dict_score = {
        'AUC': str(auc_score),
        'Precision': str(prec_score),
        'Recall': str(rec_score)
    }
    df2 = pd.DataFrame([dict_score])
    df2.to_csv(output_folder + "/binary_ml_scores_{}.csv".format(dt), sep=';', index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
