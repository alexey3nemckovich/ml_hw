import re
import spacy
import nltk.corpus
from tqdm import tqdm
import pandas as pd
from typing import List


def normalize_text(text: str):
    tm1 = re.sub('<pre>.*?</pre>', '', text, flags=re.DOTALL)
    tm2 = re.sub('<code>.*?</code>', '', tm1, flags=re.DOTALL)
    tm3 = re.sub('<[^>]+>©', '', tm2, flags=re.DOTALL)
    return tm3.replace("\n", "")


def cleanup_text(text: str, nlp: spacy.Language, stopwords: List[str]):
    punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~©'
    doc = nlp(text, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    tokens = ' '.join(tokens)
    return tokens


def preprocess_text(text: str, nlp: spacy.Language = spacy.load('en_core_web_lg'), stopwords: List[str] = nltk.corpus.stopwords.words('english')):
    text = normalize_text(text)
    text = cleanup_text(text, nlp, stopwords)
    return text


def preprocess_texts(data: pd.DataFrame, column: str, preprocessed_column: str):
    data_copy = data.copy(deep=True)

    nlp = spacy.load('en_core_web_lg')
    stopwords = nltk.corpus.stopwords.words('english')

    tqdm.pandas()
    data_copy[preprocessed_column] = data_copy[column].progress_apply(lambda x: preprocess_text(x, nlp, stopwords))
    return data_copy
