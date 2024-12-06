from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score, explained_variance_score, make_scorer, mean_absolute_error
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


# todo: change intervals for predictions
def rounded(y_pred):
    def custom_round(x):
        if x <= 0.66:
            return 0
        elif x <= 1.32:
            return 1
        else:
            return 2

    custom_round_vec = np.vectorize(custom_round)
    return custom_round_vec(y_pred)


def dt(X_train, y_train, X_test, y_test, X, y):
    clf_dt = DecisionTreeRegressor()
    clf_dt.fit(X_train, y_train)
    y_pred_dt = clf_dt.predict(X_test)
    y_pred_dt = rounded(y_pred_dt)

    print('\n\nResults for DecisionTreeRegressor')
    precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
    print("Precision: {:.2f}%".format(precision_dt * 100))

    recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
    print("Recall: {:.2f}%".format(recall_dt * 100))

    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    print("Accuracy: {:.2f}%".format(accuracy_dt * 100))

    # rounded_accuracy_scorer = make_scorer(rounded_accuracy)
    # cv_results_dt = cross_val_score(clf_dt, X, y, cv=5, scoring=rounded_accuracy_scorer)
    # print("Cross-validation results: {:.2f}% (+/- {:.2f}%)".format(cv_results_dt.mean() * 100, cv_results_dt.std() * 100))

    return clf_dt, y_pred_dt, precision_dt, recall_dt, accuracy_dt, cv_results_dt


def rf_old(X_train, y_train, X_test, y_test, X, y):
    clf_rf = RandomForestRegressor()
    clf_rf.fit(X_train, y_train)
    y_pred_rf = clf_rf.predict(X_test)
    y_pred_rf = rounded(y_pred_rf)

    print('\n\nResults for RandomForestRegressor')
    precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
    print("Precision: {:.2f}%".format(precision_rf * 100))

    recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
    print("Recall: {:.2f}%".format(recall_rf * 100))

    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print("Accuracy: {:.2f}%".format(accuracy_rf * 100))

    # rounded_accuracy_scorer = make_scorer(rounded_accuracy)
    # cv_results_rf = cross_val_score(clf_rf, X, y, cv=5, scoring=rounded_accuracy_scorer)
    # print("Cross-validation results: {:.2f}% (+/- {:.2f}%)".format(cv_results_rf.mean() * 100, cv_results_rf.std() * 100))

    return clf_rf, y_pred_rf, precision_rf, recall_rf, accuracy_rf#, cv_results_rf


def rf(X_train, y_train, X_test, y_test, X, y, clf):
    if not clf:
        clf = RandomForestRegressor()
        clf.fit(X_train, y_train)
    y_pred_rf = clf.predict(X_test)
    y_pred_rf = rounded(y_pred_rf)
    
    classes = np.unique(y_test)
    for cls in classes:
        precision = precision_score(y_test, y_pred_rf, labels=[cls], average='macro')
        recall = recall_score(y_test, y_pred_rf, labels=[cls], average='macro')
        f1 = f1_score(y_test, y_pred_rf, labels=[cls], average='macro')
        support = (y_test == cls).sum()
        
        print(f'Class {cls}:')
        print(f'    Precision: {precision:.2f}')
        print(f'    Recall:    {recall:.2f}')
        print(f'    F1-Score:  {f1:.2f}')
        print(f'    Support:   {support}\n')
    
    # Calculate overall accuracy
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    print(f'Overall Accuracy: {accuracy_rf:.2f}%')

    # Calculate macro averages for precision, recall, and f1-score
    precision_macro = precision_score(y_test, y_pred_rf, average='macro')
    print(f'Precision Macro: {precision_macro:.2f}%')
    recall_macro = recall_score(y_test, y_pred_rf, average='macro')
    print(f'Recall Macro: {recall_macro:.2f}%')
    f1_macro = f1_score(y_test, y_pred_rf, average='macro')
    print(f'f1-score macro: {f1_macro:.2f}%')
    
    return clf, y_pred_rf, precision_macro, recall_macro, f1_macro, accuracy_rf 



def gradb(X_train, y_train, X_test, y_test, X, y):
    clf_gradb = GradientBoostingRegressor(random_state=42)
    clf_gradb.fit(X_train, y_train)
    y_pred_gradb = clf_gradb.predict(X_test)
    y_pred_gradb = rounded(y_pred_gradb)

    print('\n\nResults for GradientBoostingRegressor')
    precision_gradb = precision_score(y_test, y_pred_gradb, average='weighted')
    print("Precision: {:.2f}%".format(precision_gradb * 100))

    recall_gradb = recall_score(y_test, y_pred_gradb, average='weighted')
    print("Recall: {:.2f}%".format(recall_gradb * 100))

    accuracy_gradb = accuracy_score(y_test, y_pred_gradb)
    print("Accuracy: {:.2f}%".format(accuracy_gradb * 100))

    # rounded_accuracy_scorer = make_scorer(rounded_accuracy)
    # cv_results_gradb = cross_val_score(clf_gradb, X, y, cv=5, scoring=rounded_accuracy_scorer)
    # print("Cross-validation results: {:.2f}% (+/- {:.2f}%)".format(cv_results_gradb.mean() * 100, cv_results_gradb.std() * 100))

    return clf_gradb, y_pred_gradb, precision_gradb, recall_gradb, accuracy_gradb#, cv_results_gradb


def catb(X_train, y_train, X_test, y_test, X, y):
    clf_catb = CatBoostRegressor(random_state=42, verbose=0)
    clf_catb.fit(X_train, y_train)
    y_pred_catb = clf_catb.predict(X_test)
    y_pred_catb = rounded(y_pred_catb)

    print('\n\nResults for CatBoostRegressor')
    precision_catb = precision_score(y_test, y_pred_catb, average='weighted')
    print("Precision: {:.2f}%".format(precision_catb * 100))

    recall_catb = recall_score(y_test, y_pred_catb, average='weighted')
    print("Recall: {:.2f}%".format(recall_catb * 100))

    accuracy_catb = accuracy_score(y_test, y_pred_catb)
    print("Accuracy: {:.2f}%".format(accuracy_catb * 100))

    # rounded_accuracy_scorer = make_scorer(rounded_accuracy)
    # cv_results_catb = cross_val_score(clf_catb, X, y, cv=5, scoring=rounded_accuracy_scorer)
    # print("Cross-validation results: {:.2f}% (+/- {:.2f}%)".format(cv_results_catb.mean() * 100, cv_results_catb.std() * 100))

    return clf_catb, y_pred_catb, precision_catb, recall_catb, accuracy_catb# , cv_results_catb




def cv_analysis(model, X_train, y_train, X_test, y_test, X, y):
    loo = LeaveOneOut()
    loo_errors = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf = model()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        loo_errors.append(mean_squared_error(y_test, rounded(y_pred)))

    return loo_errors


def performance_analysis(model, X_train, y_train, X_test, y_test):
    '''
    Computes the Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), R-squared (R2) score, and the Explained Variance Score of the model.
    '''
    y_pred_train = rounded(model.predict(X_train))
    y_pred_test = rounded(model.predict(X_test))

    # Root Mean Squared Error (RMSE)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    print("RMSE - Train: {:.2f}, Test: {:.2f}".format(rmse_train, rmse_test))

    # Mean Absolute Error (MAE)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)
    print("MAE - Train: {:.2f}, Test: {:.2f}".format(mae_train, mae_test))

    # R-squared (R2) score
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    print("R2 Score - Train: {:.2f}, Test: {:.2f}".format(r2_train, r2_test))

    # Explained Variance Score
    evs_train = explained_variance_score(y_train, y_pred_train)
    evs_test = explained_variance_score(y_test, y_pred_test)
    print("Explained Variance Score - Train: {:.2f}, Test: {:.2f}".format(evs_train, evs_test))


def performance_analysis_classification(model, X_train, y_train, X_test, y_test):
    '''
    Computes the accuracy, precision, recall, and f1-score of the classification model.
    '''
    y_pred_train = rounded(model.predict(X_train))
    y_pred_test = rounded(model.predict(X_test))

    # Custom rounding
    def custom_round(x):
        if x <= 0.66:
            return 0
        elif x <= 1.32:
            return 1
        else:
            return 2

    custom_round_vec = np.vectorize(custom_round)
    y_pred_train_rounded = custom_round_vec(y_pred_train)
    y_pred_test_rounded = custom_round_vec(y_pred_test)

    # Accuracy
    accuracy_train = accuracy_score(y_train, y_pred_train_rounded)
    accuracy_test = accuracy_score(y_test, y_pred_test_rounded)
    print("Accuracy - Train: {:.2f}, Test: {:.2f}".format(accuracy_train, accuracy_test))

    # Precision
    precision_train = precision_score(y_train, y_pred_train_rounded, average='weighted')
    precision_test = precision_score(y_test, y_pred_test_rounded, average='weighted')
    print("Precision - Train: {:.2f}, Test: {:.2f}".format(precision_train, precision_test))

    # Recall
    recall_train = recall_score(y_train, y_pred_train_rounded, average='weighted')
    recall_test = recall_score(y_test, y_pred_test_rounded, average='weighted')
    print("Recall - Train: {:.2f}, Test: {:.2f}".format(recall_train, recall_test))

    # F1-score
    f1_train = f1_score(y_train, y_pred_train_rounded, average='weighted')
    f1_test = f1_score(y_test, y_pred_test_rounded, average='weighted')
    print("F1-score - Train: {:.2f}, Test: {:.2f}".format(f1_train, f1_test))
