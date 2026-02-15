from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess(X, y):
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    y = y.map({'yes': 1, 'no': 0}) if y.dtype == 'object' else y

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = preprocess(X, y)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test
