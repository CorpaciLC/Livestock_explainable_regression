from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from catboost import CatBoostRegressor

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_squared_error
import numpy as np


def rounded_accuracy(y_true, y_pred):
    y_pred_rounded = np.round(y_pred)
    return accuracy_score(y_true, y_pred_rounded)



def dt(X_train, y_train, X_test, y_test, X, y):
    clf_dt = DecisionTreeRegressor()
    clf_dt.fit(X_train, y_train)
    y_pred_dt = clf_dt.predict(X_test)
    y_pred_dt = np.round(y_pred_dt)

    print('\n\nResults for DecisionTreeRegressor')
    precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
    print("Precision: {:.2f}%".format(precision_dt * 100))

    recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
    print("Recall: {:.2f}%".format(recall_dt * 100))

    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print("Accuracy: {:.2f}%".format(accuracy_dt * 100))

    rounded_accuracy_scorer = make_scorer(rounded_accuracy)
    cv_results_dt = cross_val_score(clf_dt, X, y, cv=5, scoring=rounded_accuracy_scorer)
    print("Cross-validation results: {:.2f}% (+/- {:.2f}%)".format(cv_results_dt.mean() * 100, cv_results_dt.std() * 100))

    return clf_dt, y_pred_dt, precision_dt, recall_dt, accuracy_dt, cv_results_dt


def rf(X_train, y_train, X_test, y_test, X, y):
    clf_rf = RandomForestRegressor()
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    y_pred_rf = np.round(y_pred_rf)

    print('\n\nResults for RandomForestRegressor')
    precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
    print("Precision: {:.2f}%".format(precision_rf * 100))

    recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
    print("Recall: {:.2f}%".format(recall_rf * 100))

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print("Accuracy: {:.2f}%".format(accuracy_rf * 100))

    rounded_accuracy_scorer = make_scorer(rounded_accuracy)
    cv_results_rf = cross_val_score(clf_rf, X, y, cv=5, scoring=rounded_accuracy_scorer)
    print("Cross-validation results: {:.2f}% (+/- {:.2f}%)".format(cv_results_rf.mean() * 100, cv_results_rf.std() * 100))

    return clf_rf, y_pred_rf, precision_rf, recall_rf, accuracy_rf, cv_results_rf


def gradb(X_train, y_train, X_test, y_test, X, y):
    clf_gradb = GradientBoostingRegressor(random_state=42)
    clf_gradb.fit(X_train, y_train)
    y_pred_gradb = clf_gradb.predict(X_test)
    y_pred_gradb = np.round(y_pred_gradb)

    print('\n\nResults for GradientBoostingRegressor')
    precision_gradb = precision_score(y_test, y_pred_gradb, average='weighted')
    print("Precision: {:.2f}%".format(precision_gradb * 100))

    recall_gradb = recall_score(y_test, y_pred_gradb, average='weighted')
    print("Recall: {:.2f}%".format(recall_gradb * 100))

    accuracy_gradb = accuracy_score(y_test, y_pred_gradb)
    print("Accuracy: {:.2f}%".format(accuracy_gradb * 100))

    rounded_accuracy_scorer = make_scorer(rounded_accuracy)
    cv_results_gradb = cross_val_score(clf_gradb, X, y, cv=5, scoring=rounded_accuracy_scorer)
    print("Cross-validation results: {:.2f}% (+/- {:.2f}%)".format(cv_results_gradb.mean() * 100, cv_results_gradb.std() * 100))

    return clf_gradb, y_pred_gradb, precision_gradb, recall_gradb, accuracy_gradb, cv_results_gradb


def catb(X_train, y_train, X_test, y_test, X, y):
    clf_catb = CatBoostRegressor(random_state=42, verbose=0)
    clf_catb.fit(X_train, y_train)
    y_pred_catb = clf_catb.predict(X_test)
    y_pred_catb = np.round(y_pred_catb)

    print('\n\nResults for CatBoostRegressor')
    precision_catb = precision_score(y_test, y_pred_catb, average='weighted')
    print("Precision: {:.2f}%".format(precision_catb * 100))

    recall_catb = recall_score(y_test, y_pred_catb, average='weighted')
    print("Recall: {:.2f}%".format(recall_catb * 100))

    accuracy_catb = accuracy_score(y_test, y_pred_catb)
    print("Accuracy: {:.2f}%".format(accuracy_catb * 100))

    rounded_accuracy_scorer = make_scorer(rounded_accuracy)
    cv_results_catb = cross_val_score(clf_catb, X, y, cv=5, scoring=rounded_accuracy_scorer)
    print("Cross-validation results: {:.2f}% (+/- {:.2f}%)".format(cv_results_catb.mean() * 100, cv_results_catb.std() * 100))

    return clf_catb, y_pred_catb, precision_catb, recall_catb, accuracy_catb, cv_results_catb




def cv_analysis(model, X_train, y_train, X_test, y_test, X, y):
    loo = LeaveOneOut()
    loo_errors = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = model()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        loo_errors.append(mean_squared_error(y_test, y_pred))


    return loo_errors

