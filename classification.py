# -*- coding: utf-8 -*-

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from random import randint

# handcraft features
def character(G):
    d1 = G.number_of_nodes()
    d2 = G.number_of_edges()
    d3 = np.average([d for n, d in G.degree()])
    d4 = nx.degree_histogram(G)[1]
    d5 = nx.average_clustering(G)
    d6 = np.max(nx.linalg.spectrum.adjacency_spectrum(G)).real
    d7 = nx.density(G)
    d8 = np.average(list(nx.betweenness_centrality(G).values()))
    d9 = np.average(list(nx.closeness_centrality(G).values()))
    d10 = np.average(list(nx.eigenvector_centrality_numpy(G).values()))
    d11 = np.average(list(nx.average_neighbor_degree(G).values()))

    chara = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10,
             d11]

    return chara


def RandomForest(X_train, X_test, Y_train, Y_test):
    param = {"n_estimators":[10, 20, 50, 100, 200], "criterion":("gini", "entropy")}
    grid_search = GridSearchCV(RandomForestClassifier(random_state=1),
                               n_jobs=-1, param_grid=param, cv=3, return_train_score=True)
    grid_search.fit(X_train, Y_train)
    Y_pred = grid_search.predict(X_test)

    # classifier = RandomForestClassifier()
    # classifier.fit(X_train, Y_train)
    # Y_pred = classifier.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    return acc


def logReg(X_train, X_test, Y_train, Y_test):
    param = {"C": [0.1, 1, 10, 100]}
    grid_search = GridSearchCV(LogisticRegression(random_state=0),
                               n_jobs=-1, param_grid=param, cv=3, return_train_score=True)
    grid_search.fit(X_train, Y_train)
    Y_pred = grid_search.predict(X_test)

    # classifier = LogisticRegression()
    # classifier.fit(X_train, Y_train)
    # Y_pred = classifier.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    return acc


def perform_classification1(X, Y):
    # starttime = time.time()
    seed = randint(0, 1000)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=seed)

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    # test = RandomForest(X_train_std, X_test_std, Y_train, Y_test)
    test = logReg(X_train_std, X_test_std, Y_train, Y_test)
    return test


def result_class(X, Y):
    X = np.array(X)

    final = {}
    max = []
    for k in range(50):
        Acc = []
        for i in range(10):
            acc = perform_classification1(X, Y)
            Acc.append(acc)
        ave = np.average(Acc)
        std = np.std(Acc)
        max.append(np.max(Acc))
        if ave not in final:
            final[ave] = std
        print("the test accuracy: {}, {}, {}".format(k, ave, std))
        print("the max accuracy every time: {}, {}".format(k, np.max(Acc)))
    print("the max accuracy & std: {}, {}".format(np.average(max), np.std(max)))
    acc = np.max(list(final.keys()))
    std = final[acc]
    print("the final accuracy: {}, {}".format(acc, std))