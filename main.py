import pandas as pd
from src.preprocessing import preprocess_text

def load_data():
  df_fake=pd.read_csv("data/Fake.csv")
  df_true=pd.read_csv("data/True.csv")

  # labeling fake/real news
  df_fake["label"]=0
  df_true["label"]=1

  #combining the datasets, shuffling them randomly, and resetting the index
  # to prepare a clean, randomized dataset for training a machine learning model.
  df=pd.concat([df_fake,df_true]).sample(frac=1).reset_index(drop=True)
  return df


def preprocess_dataset(df):
  df['clean_text_final'] = df['text'].apply(preprocess_text)
  return df


def main():
  # Load dataset
  df = load_data()
  print("Dataset info:")
  print(df.info())
  print("Label distribution:")
  print(df['label'].value_counts())

  # Preprocess
  df = preprocess_dataset(df)
  print("Sample preprocessed text:")
  print(df[['text', 'clean_text_final']].head())


if __name__=="__main__":
  main()