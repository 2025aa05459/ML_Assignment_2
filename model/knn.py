from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess(X, y):
    for col in X.select_dtypes(include='object').columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    y = y.map({'yes': 1, 'no': 0}) if y.dtype == 'object' else y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def train_knn(X, y):
    X_train, X_test, y_train, y_test = preprocess(X, y)

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    return model, X_test, y_test
