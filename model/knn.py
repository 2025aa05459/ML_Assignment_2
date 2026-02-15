from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

def train_knn(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train, y_train)

    return model, X_test, y_test
