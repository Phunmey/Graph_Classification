from time import time
import sys
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import shap
import random
from collections import Counter

random.seed(10)

# plt.rcParams.update({'figure.figsize': (12.0, 12.0)}) #set plotsize for all figures to be drawn
# plt.rcParams.update({'font.size': 10}) #set plot font_size for figures
def data_csv(datapath):
    #statistics = "C:/Code/src/Mapper_for_TDA/statistics.csv"
    merged_df = pd.read_csv(datapath, sep=",")
    merged_df = merged_df.drop(["Index", "graphlabel", "motif1", "motif2"], axis=1)

    X = merged_df[merged_df.columns[1:]].apply(pd.to_numeric)
    X = X.fillna(0)  # Features
    y = merged_df['dataset']  # Labels

    return(X, y)

def RF_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=300).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # model evaluation
    print("Random Forest Accuracy: ", metrics.accuracy_score(y_test, y_pred))
    accuracies = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10)
    print("Random Forest Cross Validation: ", accuracies.mean())
    print("Random Forest ROC_AUC_Score: ", roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr'))
#   plot_confusion_matrix(clf, X_test, y_test, display_labels=features, cmap=plt.cm.Blues, xticks_rotation='vertical' )
# plt.show()
    return X_test, y_test, clf

def vanilla_FI(X, X_test, y_test, clf):
    # feature importance built-in the RandomForest algorithm (Mean Decrease in Impurity)
    features = X.columns.values
    feature_importance = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    forest_importances = pd.Series(feature_importance, index=features)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig('../../results/figures/FI_MDI2.png')
    #
    # #feature importance computed with permutation method
    perm_importance = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
    perm_importances = pd.Series(perm_importance.importances_mean, index=features)

    fig, ax = plt.subplots()
    perm_importances.plot.bar(yerr=perm_importance.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.savefig('../../results/figures/FI_permutation2.png')
    #return features


def shap_vals(X, clf, X_test):
# feature importance computed with SHAP_values (Global Interpretability) (bar plot)
    plt.clf()
    features = X.columns.values
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test[:5])
    # print(explainer.expected_value) #the values obtained here serve as the base_values
    class_names = ["ENZYMES", "BZR", "COX2", "DHFR", "MUTAG", "NCI1", "PROTEINS", "REDDIT-MULTI-5K", "REDDIT-MULTI-12K"]
    shap.summary_plot(shap_values, X_test, feature_names=features, class_names=class_names, show=False,
                      plot_size=(16, 10))
    plt.savefig('../../results/figures/plot_shapvalues2.png')
# shap.summary_plot(shap_values[i (for i= 0, 1 ..., n)], X_test, feature_names=features, show=False) #for dot plot. This can be generated for a single observation(or class) at a time


def main():
    X, y = data_csv(datapath)
    X_test, y_test, clf = RF_model(X, y)
    vanilla_FI(X, X_test, y_test, clf)
    shap_vals(X, clf, X_test)

if __name__ == '__main__':
    datapath = sys.argv[1]
    main()


