from typing import List
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import re
from nltk.corpus import stopwords

class preprocessing():
    
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data

    def cleaning(self):
        def preparing_text(df: pd.DataFrame, stop_words: List[str]) -> pd.DataFrame:

            df["verified_reviews"] = df["verified_reviews"].apply(lambda x: " ".join(x.lower() for x in x.split()))

            # Replacing the special characters

            df["verified_reviews"] = df["verified_reviews"].apply(lambda x: " ".join([re.sub("[^A-Za-z0-9]+","",element) for element in str(x).split(" ")]))

            # replacing the digit/numbers

            df["verified_reviews"] = df["verified_reviews"].str.replace('d', '')

            # removing stopwords 

            df["verified_reviews"] = df["verified_reviews"].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))
            
            # Encoding target column

            df["feedback"] = LabelEncoder().fit_transform(df["feedback"])

            return df

        def encoding(data_modified: pd.DataFrame):
            token_p = Tokenizer(num_words=500, split=' ')
            token_p.fit_on_texts(data_modified["verified_reviews"].values)
            return token_p

        stop_words = stopwords.words('english')

        self.data = preparing_text(self.data, stop_words)
        token = encoding(self.data)

        return token 

def prediction(data: List[str]) -> int:
    
    with open("amazon_alexa.tsv",encoding="utf8") as file:
        data = pd.read_csv(file,sep="\t")

    clas = preprocessing(data)
    token = clas.cleaning()

    X = token.texts_to_sequences(data)
    # X = pad_sequences(X)

    print(X)

prediction("Hi world")