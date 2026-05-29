"""
Credit card customer segmentation - main analysis script.

This script does the whole pipeline in one go:
  1. load and clean the data
  2. a bit of EDA (distributions + correlation)
  3. transform + scale the features
  4. K-Means clustering (with elbow + silhouette to choose K)
  5. a second clustering method (Agglomerative) to cross-check the groups
  6. profile the clusters and project them with PCA
  7. train two classifiers (Decision Tree + Random Forest) to predict the cluster

I usually work in the notebook, but I keep this script so the whole thing
can be re-run end to end and all the figures get saved again.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                             confusion_matrix, classification_report)

# paths (relative to project root)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "credit_card_customers.csv")
FIG = os.path.join(ROOT, "figures")

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


def save(name):
    """small helper so every figure is saved the same way."""
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, name), dpi=120, bbox_inches="tight")
    plt.close()
    print("  saved figure:", name)


# ---------------------------------------------------------------------------
# 1. Load + clean
# ---------------------------------------------------------------------------
print("1. loading data ...")
df = pd.read_csv(DATA)
print("   shape:", df.shape)

# CUST_ID is just an id, it has no information for clustering
df = df.drop(columns=["CUST_ID"])

# two columns have missing values. both are money columns and are right
# skewed, so I fill with the median (mean would be pulled up by the big values)
for col in ["MINIMUM_PAYMENTS", "CREDIT_LIMIT"]:
    n_missing = df[col].isna().sum()
    df[col] = df[col].fillna(df[col].median())
    print(f"   filled {n_missing} missing values in {col} with median")

features = df.columns.tolist()
print("   features used:", len(features))


# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
print("2. EDA ...")

# distribution of 4 important money variables
key_vars = ["BALANCE", "PURCHASES", "CASH_ADVANCE", "CREDIT_LIMIT"]
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, col in zip(axes.ravel(), key_vars):
    sns.histplot(df[col], bins=50, ax=ax, color="#3b6ea5")
    ax.set_title(col)
fig.suptitle("Distribution of key financial features (all right-skewed)", fontsize=13)
save("fig1_distributions.png")

# correlation heatmap
plt.figure(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
            square=True, linewidths=.4, cbar_kws={"shrink": .7})
plt.title("Correlation between behavioural features")
save("fig2_correlation.png")


# ---------------------------------------------------------------------------
# 3. Transform + scale
# ---------------------------------------------------------------------------
print("3. transform + scale ...")

# IMPROVEMENT over the original analysis:
# most of the money columns are heavily right-skewed, which is not great for
# a distance based method like K-Means (a few huge customers dominate).
# so before scaling I apply a log1p transform to the skewed columns. log1p
# is log(1+x) so it also handles the zeros. then I z-score everything.
skewed = [c for c in features if df[c].skew() > 1.0]
print("   log-transforming skewed columns:", skewed)

df_t = df.copy()
for col in skewed:
    df_t[col] = np.log1p(df_t[col])

scaler = StandardScaler()
X = scaler.fit_transform(df_t)
X = pd.DataFrame(X, columns=features)


# ---------------------------------------------------------------------------
# 4. K-Means - choose K
# ---------------------------------------------------------------------------
print("4. choosing K for K-Means ...")

wss = []
sil = []
k_range = range(2, 11)
# silhouette on a sample, it is slow on 8950 points
sample_idx = np.random.RandomState(RANDOM_STATE).choice(len(X), 2000, replace=False)

for k in k_range:
    km = KMeans(n_clusters=k, n_init=25, max_iter=300, random_state=RANDOM_STATE)
    labels = km.fit_predict(X)
    wss.append(km.inertia_)
    sil.append(silhouette_score(X.iloc[sample_idx], labels[sample_idx]))
    print(f"   k={k}: wss={km.inertia_:.0f}  silhouette={sil[-1]:.3f}")

# elbow plot
plt.figure(figsize=(8, 5))
plt.plot(list(k_range), wss, "o-", color="#3b6ea5")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Within-cluster sum of squares")
plt.title("Elbow method")
plt.axvline(4, color="red", ls="--", alpha=.6)
save("fig3_elbow.png")

# silhouette plot
plt.figure(figsize=(8, 5))
plt.plot(list(k_range), sil, "o-", color="#c0504d")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Average silhouette width")
plt.title("Silhouette method")
plt.axvline(4, color="red", ls="--", alpha=.6)
save("fig4_silhouette.png")

K = 4
print("   --> chose K =", K)


# ---------------------------------------------------------------------------
# 5. Fit K-Means and cross-check with a second method
# ---------------------------------------------------------------------------
print("5. fitting K-Means + cross-check ...")

kmeans = KMeans(n_clusters=K, n_init=25, max_iter=300, random_state=RANDOM_STATE)
df["cluster"] = kmeans.fit_predict(X)

# IMPROVEMENT: run a completely different clustering method on the same data
# and check how much it agrees with K-Means. if the groups are real, a second
# method should find something similar. I use a sample because hierarchical
# clustering is memory heavy on the full dataset.
agg = AgglomerativeClustering(n_clusters=K)
agg_labels = agg.fit_predict(X.iloc[sample_idx])
from sklearn.metrics import adjusted_rand_score
agree = adjusted_rand_score(df["cluster"].values[sample_idx], agg_labels)
print(f"   agreement K-Means vs Hierarchical (adjusted Rand): {agree:.3f}")

# cluster sizes
sizes = df["cluster"].value_counts().sort_index()
print("   cluster sizes:\n", sizes)

plt.figure(figsize=(7, 5))
sns.countplot(x="cluster", hue="cluster", data=df, palette="Set2",
              order=sorted(df["cluster"].unique()), legend=False)
plt.title("Number of customers in each cluster")
plt.xlabel("Cluster")
save("fig5_cluster_sizes.png")


# ---------------------------------------------------------------------------
# 6. Profile clusters + PCA projection
# ---------------------------------------------------------------------------
print("6. profiling clusters ...")

profile = df.groupby("cluster").mean(numeric_only=True).round(1)
profile["size"] = sizes
profile.to_csv(os.path.join(ROOT, "data", "cluster_profile.csv"))

# also save the full table with the cluster label, the SQL script uses this
df.to_csv(os.path.join(ROOT, "data", "customers_with_clusters.csv"), index=False)
print(profile[["BALANCE", "PURCHASES", "CASH_ADVANCE",
               "PAYMENTS", "PRC_FULL_PAYMENT", "size"]])

# PCA down to 2D just for plotting
pca = PCA(n_components=2, random_state=RANDOM_STATE)
coords = pca.fit_transform(X)
plt.figure(figsize=(9, 7))
sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=df["cluster"],
                palette="Set2", s=12, alpha=.6, legend="full")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}% var)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}% var)")
plt.title("Clusters projected onto 2 principal components")
save("fig6_pca.png")

# box plots of key variables per cluster
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, col in zip(axes.ravel(), ["BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS"]):
    sns.boxplot(x="cluster", y=col, hue="cluster", data=df, ax=ax,
                palette="Set2", showfliers=False, legend=False)
    ax.set_title(col)
fig.suptitle("Behaviour by cluster", fontsize=13)
save("fig7_boxplots.png")


# ---------------------------------------------------------------------------
# 7. Supervised: predict the cluster (Decision Tree + Random Forest)
# ---------------------------------------------------------------------------
print("7. classification ...")

X_clf = X  # scaled features
y = df["cluster"]
X_train, X_test, y_train, y_test = train_test_split(
    X_clf, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)
print(f"   train={X_train.shape[0]}  test={X_test.shape[0]}")

# model 1: Decision Tree (easy to read / explain to a business audience)
tree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=20,
                              random_state=RANDOM_STATE)
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)
tree_acc = accuracy_score(y_test, tree_pred)
tree_kappa = cohen_kappa_score(y_test, tree_pred)
print(f"   Decision Tree : accuracy={tree_acc:.4f}  kappa={tree_kappa:.3f}")

# model 2: Random Forest (usually a bit more accurate, harder to read).
# IMPROVEMENT: compare the simple model with an ensemble so I can talk about
# the accuracy vs interpretability trade-off.
forest = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE,
                                n_jobs=-1)
forest.fit(X_train, y_train)
forest_pred = forest.predict(X_test)
forest_acc = accuracy_score(y_test, forest_pred)
forest_kappa = cohen_kappa_score(y_test, forest_pred)
print(f"   Random Forest : accuracy={forest_acc:.4f}  kappa={forest_kappa:.3f}")

print("\n   Decision Tree classification report:")
print(classification_report(y_test, tree_pred))

# confusion matrix for the decision tree
cm = confusion_matrix(y_test, tree_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted cluster")
plt.ylabel("Actual cluster")
plt.title(f"Decision Tree confusion matrix (test accuracy {tree_acc*100:.1f}%)")
save("fig8_confusion.png")

# feature importance from the random forest (more stable than a single tree)
imp = pd.Series(forest.feature_importances_, index=features).sort_values()
plt.figure(figsize=(9, 7))
imp.plot(kind="barh", color="#3b6ea5")
plt.title("Feature importance (Random Forest)")
plt.xlabel("Importance")
save("fig9_feature_importance.png")

# the decision tree itself, drawn out
plt.figure(figsize=(20, 10))
plot_tree(tree, feature_names=features, class_names=[str(c) for c in sorted(y.unique())],
          filled=True, rounded=True, fontsize=8, max_depth=3)
plt.title("Decision Tree (top levels)")
save("fig10_tree.png")

# save a small results summary so the README numbers stay in sync
with open(os.path.join(ROOT, "data", "results_summary.txt"), "w") as f:
    f.write(f"K = {K}\n")
    f.write(f"cluster sizes: {dict(sizes)}\n")
    f.write(f"hierarchical agreement (adjusted Rand): {agree:.3f}\n")
    f.write(f"Decision Tree accuracy: {tree_acc:.4f}, kappa: {tree_kappa:.3f}\n")
    f.write(f"Random Forest accuracy: {forest_acc:.4f}, kappa: {forest_kappa:.3f}\n")
    f.write(f"top features: {list(imp.sort_values(ascending=False).index[:5])}\n")

print("\nDONE. figures are in /figures, summary in /data/results_summary.txt")
