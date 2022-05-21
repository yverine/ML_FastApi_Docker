# -*- coding: utf-8 -*-
import os
# import important modules
import numpy as np
import pandas as pd
# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB # classifier 
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    plot_confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
# text preprocessing modules
from string import punctuation 
# text preprocessing modules
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re #regular expression
# Download dependency
for dependency in (
    "brown",
    "names",
    "punkt",
    "omw-1.4",
    "wordnet",
    "stopwords",
    "averaged_perceptron_tagger",
    "universal_tagset",
):
  nltk.download(dependency)
    
import warnings
warnings.filterwarnings("ignore")

cwd = os.getcwd() 
os.chdir(cwd)
df = pd.read_csv(cwd +'\\models\\dataset.csv', sep='\t', header=0, index_col=None)
print(df.head())

stop_words =  stopwords.words('french')
def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word
    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text =  re.sub(r'http\S+',' link ', text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
        
    # Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
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
    return(text)

df["text_clean"] = df["text"].apply(text_cleaning)
df.head()

#split features and target from  data 
X = df["text_clean"]
y = df['label'].apply(lambda x: "0" if x== "ados" else 1).values
y = np.array(y,dtype=int)

# split data into train and validate
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=y,
)

# Create a classifier in pipeline
age_classifier = Pipeline(steps=[
                               ('pre_processing',TfidfVectorizer(lowercase=False)),
                                 ('naive_bayes',MultinomialNB())
                                 ])

# train the sentiment classifier 
age_classifier.fit(X_train,y_train)

# test model performance on valid data 
y_preds = age_classifier.predict(X_valid)

accuracy_score(y_valid,y_preds)

#save model 
import joblib 
joblib.dump(age_classifier, cwd +'\\models\\age_classifier_model_pipeline.pkl')