#!/usr/bin/env python
# coding: utf-8
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
from PIL import Image


from sklearn.model_selection import cross_val_score


#%% Configuration

no_clusters= 3
question = 'labels'

current_path = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(current_path, 'data\\')
IMAGES_PATH = os.path.join(current_path, 'images\\')

filename = DATA_PATH + f'Livestock_combined_kmeans_{no_clusters}labels.csv'
# filename = DATA_PATH + 'labelled_kmeans_data_pig_poultry_last.csv'
# df = pd.read_csv(filename, delimiter=';')

cols_to_drop = ['Q1. Average availability of internet',
                'Q2. Average level of automatization',
                'Q4_a. It is easy to access TECH on the market',
                'Q4_c. It is easy to get information on TECH and distributors']
df = pd.read_csv(filename)
df.drop(cols_to_drop, axis=1, inplace=True)


df.head()

#%%

corr = df.corr()
plt.figure(figsize=(16, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix', fontsize=16)
plt.savefig(IMAGES_PATH + f'correlation_matrix_{no_clusters}labels.png')
plt.show()

#%%
plt.figure(figsize=(10, 6))
sns.countplot(x='Q6. TECH usage', hue='labels', data=df)
plt.title('Bar Plot of Q6. TECH usage and Labels')
plt.xlabel('Q6. TECH usage')
plt.ylabel('Count')
plt.legend(title='Labels')
plt.savefig(IMAGES_PATH + f'barplot_{no_clusters}labels.png')
plt.show()


#%%
count_matrix = pd.crosstab(df['labels'], df['Q6. TECH usage'])

plt.figure(figsize=(10, 8))
sns.heatmap(count_matrix, annot=True, cmap="YlGnBu", fmt='g')  # 'fmt' is used to format numbers as integers
plt.title('Heatmap of Count Data for Q6. TECH Usage and Labels')
plt.xlabel('Q6. TECH Usage')
plt.ylabel('Labels')
plt.show()


#%%
# create confusion matrix between df['labels'] and df['Q6. TECH usage']
contingency_table = pd.crosstab(df['labels'], df['Q6. TECH usage'])
contingency_table


#%%
# Plotting the stacked bar chart
contingency_table.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stacked Bar Chart of Labels vs. Q6. TECH Usage')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.legend(title='Q6. TECH usage')
plt.savefig(IMAGES_PATH + f'stacked_bar_chart_{no_clusters}labels.png')
plt.show()

#%% Train/Test Split
df_aux = df.copy()
X = df_aux.drop(question, axis=1)
y = df_aux[question].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)



#%% ML Models
from ml_models import dt, rf, catb, gradb

clf_dt, y_pred_dt, precision_dt, recall_dt, accuracy_dt, cv_results_dt = dt(X_train, y_train, X_test, y_test, X, y)

clf_rf, y_pred_rf, precision_rf, recall_rf, accuracy_rf, cv_results_rf = rf(X_train, y_train, X_test, y_test, X, y)

clf_gradb, y_pred_gradb, precision_gradb, recall_gradb, accuracy_gradb, cv_results_gradb = gradb(X_train, y_train, X_test, y_test, X, y)

clf_catb, y_pred_catb, precision_catb, recall_catb, accuracy_catb, cv_results_catb = catb(X_train, y_train, X_test, y_test, X, y)



#%%

ml_models ={
    'dt': dt(X_train, y_train, X_test, y_test, X, y),
    'rf': rf(X_train, y_train, X_test, y_test, X, y),
    'gradb': gradb(X_train, y_train, X_test, y_test, X, y),
    'catb': catb(X_train, y_train, X_test, y_test, X, y)
}

for clf_model, clf_details in ml_models.items():
    clf = clf_details[0]
    y_pred = clf_details[1]
    precision = clf_details[2]
    recall = clf_details[3]
    accuracy = clf_details[4]
    cv_results = clf_details[5]

    IMAGES_PATH_clf = IMAGES_PATH + clf_model + '\\' + f'{no_clusters}labels\\'
    os.makedirs(IMAGES_PATH_clf, exist_ok=True)

    if not os.path.exists(IMAGES_PATH_clf + 'experiment_details.txt'):
        text = f"Model: {clf}, \nNumber Labels: {no_clusters}, \nAccuracy= {accuracy}, \nPrecision={precision}, \nRecall={recall}, \nCross-validation={cv_results}"
        with open(IMAGES_PATH_clf + 'experiment_details.txt', "w") as file:
            file.write(text)
        print(text)
    


    features = list(range(X_train.shape[1])) 
    ########################## Partial Dependence Plots  ##########################

    desired_value = 2 
    fig, ax = plt.subplots(nrows=8, ncols=3, figsize=(20, 20), constrained_layout=True)

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


    for i in range(len(features), 24):
        row, col = divmod(i, 3)
        fig.delaxes(ax[row, col])

    plt.suptitle('Partial Dependence and ICE Plots for Features', fontsize=16)
    plt.savefig(IMAGES_PATH_clf + f'PDP_ICE_{no_clusters}labels_{clf_model}.png')
    plt.show()


    ########################## LIME  ##########################

    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns,
        class_names=list(set(y)), 
        mode='regression')

    for instance_index in range(10):
        instance = X_test.iloc[instance_index]

        exp = explainer.explain_instance(
            data_row=instance,
            predict_fn=clf.predict, 
            num_features=10 
        )
        exp = explainer.explain_instance(instance.values, clf.predict, num_features=len(features))
        
        fig = plt.figure(figsize=(16, 6))
        fig = exp.as_pyplot_figure(fig)

        image_path = IMAGES_PATH_clf + f'lime_{no_clusters}labels_instance{instance_index}_{clf_model}.png'
        fig.savefig(image_path, bbox_inches='tight')
        plt.close(fig)


    ########################## Shapely  ##########################

    explainer = shap.TreeExplainer(clf)
    expl_shap_values = explainer(X_test)

    shap.initjs()

    for instance_index in range(10):
        plt.figure(figsize=(12, 4)) 
        shap_waterfall_plot = shap.plots.waterfall(expl_shap_values[instance_index], max_display=20, show=False)
        
        fig = plt.gcf()
        instance_label = f"{y_test[instance_index]}"
        title = f"Label: {instance_label}, Clusters: {no_clusters}, Model: {clf_model}"
        plt.title(title, fontsize=12)
        
        image_path = IMAGES_PATH_clf + f'shap_{no_clusters}clusters_instance{instance_index}_{clf_model}.png'
        fig.savefig(image_path, bbox_inches='tight', dpi=300)
        plt.show()




# %%
