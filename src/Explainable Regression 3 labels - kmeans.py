#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import lime
import shap
from IPython.display import display
import joblib

from ml_models import *
from utils import *
from config import *


#%% Configuration

no_clusters= 3
question = 'labels'
original_df = pd.read_csv(filename)
df = original_df.copy()


#%% Load data
X_train = pd.read_csv(DATA_PATH + 'X_train.csv')
X_test = pd.read_csv(DATA_PATH + 'X_test.csv')
y_train = pd.read_csv(DATA_PATH + 'y_train.csv').values.ravel()
y_test = pd.read_csv(DATA_PATH + 'y_test.csv').values.ravel()
X = pd.concat([X_train, X_test], axis=0)
y = y_train.tolist() + y_test.tolist()


# drop Q6
X.drop(columns=['Q6. TECH usage'], inplace=True)
X_train.drop(columns=['Q6. TECH usage'], inplace=True)
X_test.drop(columns=['Q6. TECH usage'], inplace=True)
extra_str = '_noQ6'
extra_str = extra_str + '_new_rounding'


# RandomForest Model
clf = joblib.load(f'../models/rf_model_of_choice{extra_str}.pkl')

clf, y_pred, precision, recall, f1, accuracy = rf(X_train, y_train, X_test, y_test, X, y, clf)

#%% Explainability
features = list(range(X_train.shape[1])) 

label_names = {0: "Not Ready",
                1: "Partially Ready",
                2: "Ready"}


# %%
# SHAP 
shap_explanations = {0: [], 1: [], 2: []}
explainer = shap.Explainer(clf, X_train, seed=42)

for instance_index in range(len(X_test)):
    instance = X_test.iloc[instance_index:instance_index+1]  # shap requires DataFrame
    shap_values = explainer(instance)
    shap_explanations[y_test[instance_index]].append(shap_values)

feature_names = X_train.columns
aggregated_shap_explanations = aggregate_shap_values(shap_explanations, feature_names)
plot_class_explanations(aggregated_shap_explanations, 'SHAP', label_names, extra_str)
plot_aggregated_class_explanations_horiz_ordered_by_sum_of_absolutes(aggregated_shap_explanations, 'SHAP', label_names, extra_str)

#%% LIME with linear surrogate 
lime_explanations = {0:[],
                     1:[],
                     2:[]}

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=list(set(y)), 
    mode='regression',
    random_state=42)

for instance_index in range(len(X_test)):
    instance = X_test.iloc[instance_index]

    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=clf.predict, 
            num_features=10
    )
    exp = explainer.explain_instance(instance.values, clf.predict, num_features=len(features))
    lime_explanations[y_test[instance_index]].append(exp)


aggregated_explanations = aggregate_lime_explanations(lime_explanations)
plot_class_explanations(aggregated_explanations, 'LIME', label_names, extra_str)

# %% Partial Dependence Plots  
desired_value = 2
n_rows = 7
n_cols = 3
fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(20, 20), constrained_layout=True)

for i, feature in enumerate(features):
    row, col = divmod(i, n_cols)
    PartialDependenceDisplay.from_estimator(
        clf, 
        X_train, 
        [feature], 
        kind='both', 
        grid_resolution=50, 
        ax=ax[row, col],
        feature_names=X_train.columns,
        target=desired_value
    )

for i in range(len(features), 21):
    row, col = divmod(i, n_cols)
    fig.delaxes(ax[row, col])

plt.suptitle('Partial Dependence and ICE Plots for Features', fontsize=16)
plt.savefig(IMAGES_PATH + f'rf80_PDP_ICE_{no_clusters}labels_{extra_str}.png', dpi=300)
plt.savefig(IMAGES_PATH + f'rf80_PDP_ICE_{no_clusters}labels_{extra_str}.eps', dpi=300)
plt.show()


# %%
