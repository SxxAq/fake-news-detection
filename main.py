import pandas as pd
from src.preprocessing import preprocess_text
from src.features import get_bow_features, get_tfidf_features
from src.model import train_model, evaluate_models
import joblib

def load_data():
    df_fake = pd.read_csv("data/Fake.csv")
    df_true = pd.read_csv("data/True.csv")

    # labeling fake/real news
    df_fake["label"] = 0
    df_true["label"] = 1

    # combining the datasets, shuffling them randomly, and resetting the index
    # to prepare a clean, randomized dataset for training a machine learning model.
    df = pd.concat([df_fake, df_true]).sample(frac=1).reset_index(drop=True)
    return df


def preprocess_dataset(df):
    df["clean_text_final"] = df["text"].apply(preprocess_text)
    return df


def main():
    # Load dataset
    df = load_data()
    print("Dataset info:")
    print(df.info())
    print("Label distribution:")
    print(df["label"].value_counts())

    # Preprocess
    df = preprocess_dataset(df)
    print("Sample preprocessed text:")
    print(df[["text", "clean_text_final"]].head())

    # Feature Engineering
    X_tfidf, tfidf_vectorizer = get_tfidf_features(df["clean_text_final"])
    print("TF-IDF Feature Matrix Shape: ", X_tfidf.shape)

    X_bow, bow_vectorizer = get_bow_features(df["clean_text_final"])
    print("Bag of words feature matrix shape: ", X_bow.shape)

    # Model training
    y = df["label"]

    X = X_tfidf
    print("Training started...")
    models, X_train,  y_train,X_test, y_test = train_model(X, y)
    # Save Random Forest and TF-IDF vectorizer
    joblib.dump(models["RandomForest"], "models/random_forest.pkl")
    joblib.dump(tfidf_vectorizer, "models/tfidf_vectorizer.pkl")
    print("Training started...")
    evaluate_models(models, X_test, y_test)


if __name__ == "__main__":
    main()
