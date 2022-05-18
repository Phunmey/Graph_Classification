import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
import shap
import random

random.seed(10)
plt.rcParams.update({'figure.figsize': (16.0, 14.0)})  # set plotsize for all figures to be drawn
plt.rcParams.update({'font.size': 12})  # set plot font_size for figures

statistics = "C:/Code/src/Mapper_for_TDA/statistics.csv"
merged_df = pd.read_csv(statistics, sep=",")
merged_df = merged_df.drop(["Index", "graphlabel", "motif1", "motif2"], axis=1)

X = merged_df[merged_df.columns[1:]].apply(pd.to_numeric)
X = X.fillna(0)  # Features
y = merged_df['dataset']  # Labels to be predicted

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

######################
# random forest
clf = RandomForestClassifier(n_estimators=300).fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Random Forest Accuracy: ", metrics.accuracy_score(y_test, y_pred))

explainer = shap.Explainer(clf)
shap_values = explainer.shap_values(X_test[:5])

feature_names = X_test.columns.values #list the feature names
label_names = (np.unique(np.array(y))).tolist()
# for i in range(len(shap_values)):
#     new_shap = (pd.DataFrame(shap_values[i], columns=feature_names)).agg(['mean', 'std'], axis=0)
#     u = new_shap

new_shap = (pd.DataFrame(np.concatenate(shap_values), columns=feature_names)).round(3)
lol = len(X_test[:5])
data_name = [name for name in label_names for i in range(lol)]
new_shap.insert(0, 'dataset', data_name)
mean_std = new_shap.groupby('dataset').agg(['mean', 'std'], axis=0)
#new_shap.insert(0, 'dataset', )
#new_shap.to_csv("C:/Code/src/Shapely_for_TDA/shapley.csv")


feature_loc = [X_test.columns.get_loc(column) for column in feature_names] #obtain the location of the features from the data used in Shape Explainer

#
#
mean_ = []
# std_ = []
# for i in range(len(shap_values)): #obtain a class from the list of shap_values
#     mean_vals = np.abs(shap_values[i][:, feature_loc]).mean(axis=0) # Compute mean shap values per class
#     std_vals = np.abs(shap_values[i][:, feature_loc]).std(axis=0)  # Compute std shap values per class
#     mean_.append(mean_vals)
#     std_.append(std_vals)
#
#
# df_mean = pd.DataFrame(mean_, columns=feature_names)
# df_mean.index = label_names
# df_mean.round(decimals=3)
# df_std = pd.DataFrame(std_, columns=feature_names)
# df_std.index = label_names
# df_std.round(decimals=3)

# df_mean.to_csv("C:/Code/src/Shapely_for_TDA/shapley_mean.csv")
# df_std.to_csv("C:/Code/src/Shapely_for_TDA/shapley_std.csv")

# shap_importance = pd.DataFrame(list(zip(class_names, feature_names, mean_vals, std_vals)),
#                                columns=['cname','feature_name', 'mean_of_feature', 'std_of_feature'])
# shap_importance.sort_values(by=['mean_of_feature'],
#                             ascending=False, inplace=True)
# print([shap_importance.head(), shap_importance.shape])

# SHAPELY
# Variable importance plot - Global interpretability
# features = list(X.columns)
# explainer = shap.TreeExplainer(clf)
# shap_values = explainer.shap_values(X_test[:10]) #this has the shape (9, 4914, 11)
# class_names = ["ENZYMES", "BZR", "COX2", "DHFR", "MUTAG", "NCI1", "PROTEINS", "REDDIT-MULTI-5K", "REDDIT-MULTI-12K"]
# shap.summary_plot(shap_values, X_test, feature_names=feature_names, class_names=class_names) #indexed to obbtain shap values of TRUE predictions
# # plt.savefig('../../results/figures/shapplot3.png')
#shap.summary_plot(shap_values[0], X_test, feature_names, class_names, plot_type='bar')

# explainer = shap.TreeExplainer(clf)
# shap_values = explainer.shap_values(X_train)
# fig_shap = shap.summary_plot(shap_values, features=X_train, feature_names=features, plot_type='bar', show=False,
#                              plot_size=(16, 10))
# fig = shap_values.plot(bar_width = 16)
# plt.savefig('../../results/figures/shapplot2.png')
