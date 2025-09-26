import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def train_classifier(X_train, y_train, n_estimators=200):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    clf.fit(X_train, y_train)
    return clf

def train_regressor(X_train, y_train, n_estimators=200):
    reg = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    reg.fit(X_train, y_train)
    return reg
