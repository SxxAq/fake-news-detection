from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def train_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    models = {}

    # logistic regression
    lr = LogisticRegression(max_iter=1000,solver='saga')
    lr.fit(X_train, y_train)
    models["LogisticRegression"] = lr

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    models["NaiveBayes"] = nb

    # random forest
    rf = RandomForestClassifier(n_estimators=50,n_jobs=-1, random_state=random_state)
    rf.fit(X_train, y_train)
    models["RandomForest"] = rf

    return models, X_train, y_train, X_test, y_test


def evaluate_models(models, X_test, y_test):
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"======{name}======")
        print("Accuracy: ", accuracy_score(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\n")
