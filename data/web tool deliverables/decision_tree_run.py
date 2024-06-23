#%%
!pip install scikit-learn==0.24.2
!pip install joblib==1.0.1

#%%
import pandas as pd
# import pickle
# from sklearn.utils.validation import check_is_fitted
# import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
import dice_ml
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import tree
import pandas as pd

data_path = r'.\final_labelled_data.csv'
model_path = r'.\decision_tree.sav'
data = pd.read_csv(data_path, index_col=False)

data.head()

#%%
question = 'labels'
X = data.drop(question, axis=1).copy()
y = data[question].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf1 = tree.DecisionTreeClassifier(criterion='entropy')#,  min_samples_leaf=3, max_depth=7)
clf2 = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
scores1 = cross_val_score(clf1, X, y, cv=5)
print('Full tree: {},\n mean:{:.2f}, s.d.:{:.2f}'.format(scores1, scores1.mean(), scores1.std()))

scores2 = cross_val_score(clf2, X, y, cv=5)
print('Pruned tree: {},\n mean:{:.2f}, s.d.:{:.2f}'.format(scores2, scores2.mean(), scores2.std()))

model = clf1

model.fit(X_train, y_train)


# # model = pickle.load(open(model_path, 'rb'))
# model = joblib.load(model_path)
# check_is_fitted(model)
# result = model.score(X, y)
# print(result)



#%%
model.predict(X_test)




# %%
# Define features for which to create PDP and ICE plots
features = list(range(X.shape[1]))  # Indices of all features

# Plot partial dependence and ICE plots
fig, ax = plt.subplots(nrows=8, ncols=3, figsize=(20, 20), constrained_layout=True)

# Loop through the features and create subplots
for i, feature in enumerate(features):
    row, col = divmod(i, 3)
    PartialDependenceDisplay.from_estimator(
        model, 
        X_train, 
        [feature], 
        kind='both', 
        grid_resolution=50, 
        ax=ax[row, col],
        feature_names=X.columns
    )
    ax[row, col].set_title(f'Feature: {X.columns[feature]}')

plt.suptitle('Partial Dependence and ICE Plots for Features', fontsize=16)
plt.show()

# Initialize DiCE
data_dice = dice_ml.Data(dataframe=pd.concat([X_train, y_train], axis=1), continuous_features=X.columns.tolist(), outcome_name='labels')
model_dice = dice_ml.Model(model=model, backend="sklearn")

# Generate counterfactuals
dice = dice_ml.Dice(data_dice, model_dice)
query_instance = X_test.iloc[0]#.to_dict()  # Example instance
desired_target_value = desired_value  # Desired target value to match the labels

cf = dice.generate_counterfactuals(query_instance, total_CFs=5, desired_class=[desired_target_value, desired_target_value])

# Visualize counterfactuals
cf.visualize_as_dataframe()
# %%
