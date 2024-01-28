import pandas as pd
import numpy as np
import sklearn
import accessodati as acv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
import DEFINITIVO_GIOCATORI as dg
from sklearn.metrics import mean_squared_error

#Se ho un dataframe lo converto in pickle con dataframe.to_pickle("nome del file pkl")
#df.to_pickle('gca_pass_shot.pkl')
#Una volta salvato il file pickle su pycharm me lo posso leggere con panda

# "Player","Goals","Assists","Keypasses","xGChain","xGBuildup", Yearly Salary,Age,Nationality
df_giocatori=pd.concat([dg.df_giocatori_seriea,dg.df_giocatori_premier])
#print(df_forward)
X=df_giocatori.iloc[:,[1,2,3,4,5,7]].values.reshape(-1,6)
median_salary = np.median(np.array(df_giocatori.iloc[:, 6]))
target = np.where(df_giocatori.iloc[:, 6] > median_salary, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X, target,test_size=0.33,random_state=0)
clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train,y_train)
"""
scores=[]
for i in range(0,100):
      X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=i)
      estimatore=DecisionTreeRegressor()
      estimatore.fit(X_train,y_train)
      scores.append(estimatore.score(X_test,y_test))

plt.plot(range(0,100),scores)
plt.ylabel("Score")
#plt.show()
"""
n_nodes = clf.tree_.node_count
children_left = clf.tree_.children_left
children_right = clf.tree_.children_right
feature = clf.tree_.feature
n_features=clf.tree_.n_features
threshold = clf.tree_.threshold

node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
is_leaves = np.zeros(shape=n_nodes, dtype=bool)
stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
while len(stack) > 0:
    # `pop` ensures each node is only visited once
    node_id, depth = stack.pop()
    node_depth[node_id] = depth

    # If the left and right child of a node is not the same we have a split
    # node
    is_split_node = children_left[node_id] != children_right[node_id]
    # If a split node, append left and right children and depth to `stack`
    # so we can loop through them
    if is_split_node:
        stack.append((children_left[node_id], depth + 1))
        stack.append((children_right[node_id], depth + 1))
    else:
        is_leaves[node_id] = True

print(
    "The binary tree structure has {n} nodes and has "
    "the following tree structure:\n".format(n=n_nodes)
)
for i in range(n_nodes):
    if is_leaves[i]:
        print(
            "{space}node={node} is a leaf node.".format(
                space=node_depth[i] * "\t", node=i
            )
        )
    else:
        print(
            "{space}node={node} is a split node: "
            "go to node {left} if X[:, {feature}] <= {threshold} "
            "else to node {right}.".format(
                space=node_depth[i] * "\t",
                node=i,
                left=children_left[i],
                feature=feature[i],
                threshold=threshold[i],
                right=children_right[i],
            )
        )
tree.plot_tree(clf)
#plt.show()

path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
plt.show()
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
#print(train_scores)
#print(test_scores)
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("R2")
ax.set_title("R2 vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()
print(clf.score(X_test,y_test))
print(mean_squared_error(y_test,clf.predict(X_test)))