from pandas import DataFrame
from sklearn.model_selection import train_test_split


def prepare_database(database: DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        database.iloc[:,:-1],
        database.iloc[:,-1],
        test_size=0.3,
        random_state=5
    )

    return X_train, X_test, y_train, y_test


def filter_database(df: DataFrame, active_columns: str):
    cols = [ x for x in range(len(active_columns)) if active_columns[x] == '0' ]

    return df.drop(df.columns[cols], axis=1)
