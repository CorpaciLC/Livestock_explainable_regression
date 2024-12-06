#%%
!pip install gapstatistics


#%%

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn import metrics
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans

from pycvi.cluster import get_clustering

from ml_models import *
from config import *
import cvi

#%% Configuration

no_clusters= 3
question = 'labels'
original_df = pd.read_csv(filename)
df = original_df.copy()


#%%
X_train = pd.read_csv(DATA_PATH + 'X_train.csv')
X_test = pd.read_csv(DATA_PATH + 'X_test.csv')
y_train = pd.read_csv(DATA_PATH + 'y_train.csv').values.ravel()
y_test = pd.read_csv(DATA_PATH + 'y_test.csv').values.ravel()
X = pd.concat([X_train, X_test], axis=0)
y = y_train.tolist() + y_test.tolist()

# Reset the indices of the DataFrames
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

# Concatenate the DataFrames
X = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)
y = pd.Series(y_train.tolist() + y_test.tolist()).reset_index(drop=True)

# Verify the indices are unique
print(X.index.is_unique)  
print(y.index.is_unique)  

# drop Q6
X.drop(columns=['Q6. TECH usage'], inplace=True)
X_train.drop(columns=['Q6. TECH usage'], inplace=True)
X_test.drop(columns=['Q6. TECH usage'], inplace=True)


#%%

part_separation_values = []
rCIP_values = []
WB_values = []

max_clusters = 6

for no_clusters in range(2, max_clusters):
    if no_clusters == 3:
        labeledData_c = original_df.copy()
        labels_c = y
    else:
        km = KMeans(n_clusters=no_clusters, random_state=42)
        clusters = km.fit_predict(X)
        
        labels_c = pd.DataFrame(clusters)
        labeledData_c = pd.concat((X, labels_c), axis=1)
        labeledData_c = labeledData_c.rename({0: 'labels'}, axis=1)

    rCIP = cvi.rCIP()
    rCIP_values = np.append(rCIP_values, rCIP.get_cvi(X.values, labels_c.values.ravel()))

    WBv = cvi.WB()
    WB_values = np.append(WB_values, WBv.get_cvi(X.values, labels_c.values.ravel()))



plt.figure()
plt.plot(range(2, max_clusters), rCIP_values, marker='o')
plt.title('Renyi\'s representative Cross Information Potential ', fontsize=18)
plt.xlabel('Number of Clusters', fontsize=18)
plt.ylabel('rCIP', fontsize=18)
plt.yticks(fontsize=18)  
plt.xticks(range(2, max_clusters), fontsize=18)
plt.grid(True)
plt.savefig('../images/rCIP.eps', format='eps', dpi=300)

plt.figure()
plt.plot(range(2, max_clusters), WB_values, marker='o')
plt.title('WB-index', fontsize=18)
plt.xlabel('Number of Clusters', fontsize=18)
plt.ylabel('WB-index', fontsize=18)
plt.yticks(fontsize=18)  
plt.xticks(range(2, max_clusters), fontsize=18)
plt.grid(True)
plt.savefig('../images/wb_index.eps', format='eps', dpi=300)



#%%