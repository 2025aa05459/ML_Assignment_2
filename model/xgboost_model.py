from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

def train_xgboost(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    return model, X_test, y_test
