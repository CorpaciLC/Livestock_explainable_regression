#%%

import os

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_validate
from sklearn import tree


from sklearn.inspection import PartialDependenceDisplay
import lime
from lime import lime_tabular
import shap
from IPython.display import display

from selenium import webdriver
from sklearn.cluster import KMeans
from PIL import Image

import itertools
import joblib


from sklearn.model_selection import cross_val_score


#%% Configuration

no_clusters= 3
question = 'labels'

current_path = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(current_path, 'data\\')
IMAGES_PATH = os.path.join(current_path, 'images\\')

filename = DATA_PATH + f'Livestock_combined_kmeans_{no_clusters}labels.csv'
original_df = pd.read_csv(filename)
df = original_df.copy()


#%% PCA

question = 'labels'
import plotly.express as px
from sklearn.decomposition import PCA, KernelPCA, FastICA

aux_data = df.drop(columns=[question], axis=1)

pca = PCA(n_components=4)
components = pca.fit_transform(aux_data)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}
fig = px.scatter_matrix(
    components,
    labels=labels,
    dimensions=range(4),
    color=df[question],
    title='PCA:  {} - {} '.format(no_clusters, question),
    color_continuous_scale=px.colors.diverging.RdYlBu
)
fig.update_traces(diagonal_visible=False)
fig.show()


pca_data = pca.fit_transform(df)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=df['labels'], cmap="RdYlBu")
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('4D PCA Plot')
fig.colorbar(scatter, ax=ax)


plt.show()


#%% Setting labels based on PC1

# df['PC1'] = components[:, 0]
# values = df.groupby('labels')['PC1'].mean().sort_values()

# label_map = {label: i for i, label in enumerate(values.index)}
# df['labels'] = df['labels'].map(label_map)
# df.drop('PC1', axis=1, inplace=True)
# df


# #%% Correlation Matrix

# corr = df.corr()
# plt.figure(figsize=(16, 6))
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
# plt.title('Correlation Matrix', fontsize=16)
# plt.savefig(IMAGES_PATH + f'correlation_matrix_{no_clusters}labels.png')
# plt.show()


#%% Bar plot of Q6. and Labels
plt.figure(figsize=(10, 6))
sns.countplot(x='Q6. TECH usage', hue='labels', data=df)
plt.title('Bar Plot of Q6. TECH usage and Labels')
plt.xlabel('Q6. TECH usage')
plt.ylabel('Count')
plt.legend(title='Labels')
plt.savefig(IMAGES_PATH + f'barplot_{no_clusters}labels.png')
plt.show()


#%% Crosstab Heatmap
count_matrix = pd.crosstab(df['labels'], df['Q6. TECH usage'])
plt.figure(figsize=(10, 8))
sns.heatmap(count_matrix, annot=True, cmap="YlGnBu", fmt='g')  # 'fmt' formats numbers as integers
plt.title('Heatmap of Count Data for Q6. TECH Usage and Labels')
plt.xlabel('Q6. TECH Usage')
plt.ylabel('Labels')
plt.savefig(IMAGES_PATH + 'cross_tab_heatmap.png')
plt.show()


#%% confusion matrix between df['labels'] and df['Q6. TECH usage']
contingency_table = pd.crosstab(df['labels'], df['Q6. TECH usage'])
contingency_table.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stacked Bar Chart of Labels vs. Q6. TECH Usage')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.legend(title='Q6. TECH usage')
plt.savefig(IMAGES_PATH + f'stacked_bar_chart_{no_clusters}labels.png')
plt.show()



#%%
# load X_train, X_test, y_train, y_test from file
X_train = pd.read_csv(DATA_PATH + 'X_train.csv')
X_test = pd.read_csv(DATA_PATH + 'X_test.csv')
y_train = pd.read_csv(DATA_PATH + 'y_train.csv').values.ravel()
y_test = pd.read_csv(DATA_PATH + 'y_test.csv').values.ravel()
X = pd.concat([X_train, X_test], axis=0)
y = y_train.tolist() + y_test.tolist()

# clf_rf, y_pred_rf, precision_rf, recall_rf, accuracy_rf, cv_results_rf = rf(X_train, y_train, X_test, y_test, X, y)

# clf_model = 'rf'
# clf = clf_rf
# accuracy = accuracy_rf
# precision = precision_rf
# recall = recall_rf
# cv_results = cv_results_rf


# IMAGES_PATH_clf = IMAGES_PATH + clf_model + '\\' + f'{no_clusters}labels\\'
# os.makedirs(IMAGES_PATH_clf, exist_ok=True)

# if not os.path.exists(IMAGES_PATH_clf + 'experiment_details.txt'):
#     text = f"Model: {clf}, \nNumber Labels: {no_clusters}, \nAccuracy= {accuracy}, \nPrecision={precision}, \nRecall={recall}, \nCross-validation={cv_results}"
#     with open(IMAGES_PATH_clf + 'experiment_details.txt', "w") as file:
#         file.write(text)
#     print(text)

# # save model to file
# import joblib
# joblib.dump(clf, '../models/rf_model_of_choice.pkl')

# load model from file
clf = joblib.load('../models/rf_model_of_choice.pkl')

# read and print experiment_details.txt
with open(IMAGES_PATH + 'experiment_details.txt', "r") as file:
    print(file.read())



#########################################################################################
#########################################################################################
#########################################################################################
# Explainability

features = list(range(X_train.shape[1])) 

label_names = {0: "Not Ready",
                1: "Partially Ready",
                2: "Ready"}

def plot_class_explanations(aggregated_explanations, method):

    min_importance = min(min(importance.values()) for importance in aggregated_explanations.values()) * 1.1
    max_importance = max(max(importance.values()) for importance in aggregated_explanations.values()) * 1.1
    
    for cluster, feature_importances in aggregated_explanations.items():
        features = list(feature_importances.keys())
        importances = list(feature_importances.values())
        sorted_indices = np.argsort(importances)
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        
        plt.figure(figsize=(15, 10))  
        plt.barh(sorted_features, sorted_importances, color='blue', alpha=0.7)#, label=label_names[cluster])
        plt.title(f'{method} Explanations for \"{label_names[cluster]}\"', fontsize=16)
        plt.xlabel('Influence on Prediction', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.xticks(fontsize=12)  
        plt.xlim(min_importance, max_importance)
        plt.gca().invert_yaxis()  # Invert y-axis to have the largest bar at the top
        plt.legend(loc='best')
        plt.tight_layout() 

        # save image
        plt.savefig(IMAGES_PATH + f'rf80_{method}_explanations_{label_names[cluster]}.png')
        plt.show()

def plot_aggregated_class_explanations(aggregated_explanations, method):

    min_importance = min(min(importance.values()) for importance in aggregated_explanations.values()) * 1.1
    max_importance = max(max(importance.values()) for importance in aggregated_explanations.values()) * 1.1
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    colors = ['red', 'yellow', 'blue']
    
    for cluster, feature_importances in aggregated_explanations.items():
        features = list(feature_importances.keys())
        importances = list(feature_importances.values())
        sorted_indices = np.argsort(importances)
        sorted_features = [features[i] for i in sorted_indices]
        sorted_importances = [importances[i] for i in sorted_indices]
        ax.barh(sorted_features, sorted_importances, color=colors[cluster], alpha=0.7, label=label_names[cluster])
    
    ax.set_title(f'{method} Explanations', fontsize=16)
    ax.set_xlabel('Influence on Prediction', fontsize=14)
    ax.set_ylabel('Features', fontsize=14)
    # ax.set_xticks(fontsize=12)  
    ax.set_xlim(min_importance, max_importance)
    ax.invert_yaxis()  # Invert y-axis to have the largest bar at the top
    ax.legend(['"Not Ready', 'Partially Ready', 'Ready'])
    ax.legend(loc='best')
    plt.tight_layout() 
    # save image
    plt.savefig(IMAGES_PATH + f'rf80_{method}_explanations_aggregated.png')
    plt.show()

########################## LIME  ##########################

lime_explanations = {0:[],
                     1:[],
                     2:[]}

explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    class_names=list(set(y)), 
    mode='regression')

for instance_index in range(len(X_test)):
    instance = X_test.iloc[instance_index]

    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=clf.predict, 
        num_features=10 
    )
    exp = explainer.explain_instance(instance.values, clf.predict, num_features=len(features))
    lime_explanations[y_test[instance_index]].append(exp)

def aggregate_lime_explanations(lime_explanations):
    aggregated_explanations = {}
    for cluster, explanations in lime_explanations.items():
        feature_importances = {}
        for exp in explanations:
            for feature, importance in exp.as_list():
                feature_name = feature.split(' <= ')[0] 
                if feature_name in feature_importances:
                    feature_importances[feature_name] += importance
                else:
                    feature_importances[feature_name] = importance
        for feature in feature_importances:
            feature_importances[feature] /= len(explanations)
        aggregated_explanations[cluster] = feature_importances
    return aggregated_explanations

aggregated_explanations = aggregate_lime_explanations(lime_explanations)
plot_class_explanations(aggregated_explanations, 'LIME')



# %%
########################## SHAP ##########################

shap_explanations = {0: [], 1: [], 2: []}
explainer = shap.Explainer(clf, X_train)

for instance_index in range(len(X_test)):
    instance = X_test.iloc[instance_index:instance_index+1]  # shap requires DataFrame
    shap_values = explainer(instance)
    shap_explanations[y_test[instance_index]].append(shap_values)

def aggregate_shap_values(shap_explanations, feature_names):
    aggregated_explanations = {0: {}, 1: {}, 2: {}}
    
    for cluster in shap_explanations.keys():
        feature_importances = {feature: [] for feature in feature_names}
        
        for shap_values in shap_explanations[cluster]:
            for feature_index, feature_value in enumerate(shap_values.values[0]):
                feature_name = feature_names[feature_index]
                feature_importances[feature_name].append(feature_value)
        
        for feature_name in feature_importances.keys():
            aggregated_explanations[cluster][feature_name] = np.median(feature_importances[feature_name])

    return aggregated_explanations

feature_names = X_train.columns
aggregated_shap_explanations = aggregate_shap_values(shap_explanations, feature_names)
plot_class_explanations(aggregated_shap_explanations, 'SHAP')
plot_aggregated_class_explanations(aggregated_shap_explanations, 'SHAP')



# %%
########################## Partial Dependence Plots  ##########################
desired_value = 2
fig, ax = plt.subplots(nrows=7, ncols=3, figsize=(20, 20), constrained_layout=True)

for i, feature in enumerate(features):
    row, col = divmod(i, 3)
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
    row, col = divmod(i, 3)
    fig.delaxes(ax[row, col])

plt.suptitle('Partial Dependence and ICE Plots for Features', fontsize=16)
plt.savefig(IMAGES_PATH + f'rf80_PDP_ICE_{no_clusters}labels.png')
plt.show()

###################################################################################################################################
# # %% LIME with tree - incomplete
# X = np.array(X)
# y = np.array(y)

# from lime.lime_tabular import LimeTabularExplainer
# from sklearn.tree import DecisionTreeRegressor
# # Create LIME Explainers

# # Define feature names and class names for LIME
# feature_names = df.columns 
# class_names = list(set(y_train))  # Assuming y_train contains class labels

# # Ensure X_train is in the right format for LIME (NumPy array or Pandas DataFrame)
# if isinstance(X_train, pd.DataFrame):
#     X_train = X_train.values
# if isinstance(X_test, pd.DataFrame):
#     X_test = X_test.values

# # Create LIME Explainers for regression
# explainer = LimeTabularExplainer(X_train, feature_names=feature_names, discretize_continuous=True, mode='regression')

# def explain_with_linear(instance):
#     exp = explainer.explain_instance(instance, clf.predict, num_features=10)
#     return exp

# def explain_with_tree(instance):
#     exp = explainer.explain_instance(instance, clf.predict, num_features=10, model_regressor=DecisionTreeRegressor())
#     return exp

# # Explain a sample instance from X_test
# sample_instance = X_test[0]

# linear_explanation = explain_with_linear(sample_instance)
# tree_explanation = explain_with_tree(sample_instance)

# # Print explanations (or save them if needed)
# print("Linear LIME Explanation:")
# print(linear_explanation.as_list())
# print("\nTree-based LIME Explanation:")
# print(tree_explanation.as_list())

###################################################################################################################################

# %%
import numpy as np
import pandas as pd
from lime import lime_tabular
from sklearn.tree import DecisionTreeRegressor

# Ensure X_train and X_test are numpy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# Initialize the LIME explainer
explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=list(X_train.columns) if isinstance(X_train, pd.DataFrame) else [f'feature_{i}' for i in range(X_train.shape[1])],
    class_names=[str(cls) for cls in np.unique(y_train)],
    mode='regression'
)

# Initialize dictionary to store LIME explanations
lime_explanations = {0: [], 1: [], 2: []}

# Function to use a decision tree as the surrogate model
def custom_explain_instance(instance, predict_fn, num_features):
    # Use LIME's explain_instance method to get the explanation object
    explanation = explainer.explain_instance(instance, predict_fn, num_features=num_features)
    
    # Extract the perturbed data and corresponding predictions
    perturbed_data, perturbed_labels, distances = explainer.data_labels_distances(instance, predict_fn)
    
    # Fit a decision tree regressor as the surrogate model
    tree = DecisionTreeRegressor(max_depth=5)  # Adjust the max_depth as needed
    tree.fit(perturbed_data, perturbed_labels, sample_weight=distances)
    
    # Get the feature importances from the decision tree
    feature_importances = tree.feature_importances_
    feature_names = explainer.feature_names
    
    # Sort features by importance and select the top num_features
    top_features = np.argsort(feature_importances)[-num_features:]
    top_features = sorted(top_features, key=lambda x: -feature_importances[x])
    
    # Create the explanation in the same format as LIME's default output
    exp = [(feature_names[i], feature_importances[i]) for i in top_features]
    
    return exp

# Generate explanations for the test instances
for instance_index in range(len(X_test)):
    instance = X_test[instance_index]
    
    # Use the custom explain instance function
    exp = custom_explain_instance(instance, clf.predict, num_features=10)
    
    # Store the explanation in the appropriate cluster
    lime_explanations[y_test[instance_index]].append(exp)

# The lime_explanations dictionary now contains the explanations using a decision tree as the surrogate model

