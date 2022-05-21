# -*- coding: utf-8 -*-
# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re  # regular expression
import os
from os.path import dirname, join, realpath
import joblib
import uvicorn
from fastapi import FastAPI

app = FastAPI(
    title="age detection Model API",
    description="model de classification des age en fonction des textes",
    version="0.78",
)

import os
cwd = os.getcwd()

# chargement du model
with open(
    join(dirname(realpath("__file__")), cwd+"\\models\\age_classifier_model_pipeline.pkl"), "rb"
) as f:
    model = joblib.load(f)

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"http\S+", " link ", text)
    text = re.sub(r"\b\d+(?:\.\d+)?\s+", "", text)  # remove numbers
    # Remove punctuation from text
    text = "".join([c for c in text if c not in punctuation])
    # Optionally, remove stop words
    if remove_stop_words:
        # load stopwords
        stop_words = stopwords.words("english")
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    # Return a list of words
    return text

@app.get("/predict-age")
def predict_age(review: str):
   
    # clean the text
    text_clean = text_cleaning(review)
    
    # perform prediction
    prediction = model.predict([text_clean])
    output = int(prediction[0])
    probas = model.predict_proba([text_clean])
    output_probability = "{:.2f}".format(float(probas[:, output]))
    
    # output dictionary
    age = {1: "adult", 0: "ados"}
    
    # show results
    result = {"prediction": age[output], "Probability": output_probability}
    return result



