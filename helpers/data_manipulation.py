from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score as score
from sklearn.neighbors import KNeighborsRegressor


def preparate_database(database: DataFrame, active_columns: str):
    cols = [ x for x in range(len(active_columns)) if active_columns[x] == '0' ]
    cols.append(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        database.drop(database.columns[cols], axis=1),
        database['Y'],
        test_size=0.3,
        random_state=5
    )

    return X_train, X_test, y_train, y_test


def r2_score(X_train, X_test, y_train, y_test):
    classifier = KNeighborsRegressor(3).fit(X_train, y_train.values.ravel())
    y_predict = classifier.predict(X_test)

    return score(y_true=y_test, y_pred=y_predict)
