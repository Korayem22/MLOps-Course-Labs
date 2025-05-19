import sklearn

def train_RF(X_train, y_train):
    """
    Train a random forest regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        RandomForestRegressor: trained random forest regression model
    """
    n_estimators = 100
    criterion = "gini"
    max_depth = None
    prams = {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
    }
    RF = sklearn.ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
    )
    print("Training Random Forest model...")
    RF.fit(X_train, y_train)
    return RF,prams

def train_SVM(X_train,y_train):
    """
    Train a Support Vector Machine model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        SVC: trained Support Vector Machine model
    """
    kerenal = 'rbf'
    c = 1
    gamma = 0.1
    prams = {
        "kernel": kerenal,
        "C": c,
        "gamma": gamma
    }
    svc = sklearn.svm.SVC(kernel=kerenal, C=c, gamma=gamma)
    print("Training Support Vector Machine model...")
    svc.fit(X_train,y_train)
    return svc,prams

def train_GradientBoosting(X_train,y_train):
    """
    Train a Gradient Boosting model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        GradientBoostingClassifier: trained Gradient Boosting model
    """
    n_estimators = 100
    learning_rate = 0.1
    prams = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate
    }
    gb = sklearn.ensemble.GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    print("Training Gradient Boosting model...")
    gb.fit(X_train,y_train)
    return gb,prams