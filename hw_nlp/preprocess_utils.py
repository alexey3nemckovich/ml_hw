import re
from typing import List

import nltk.corpus
import pandas as pd
import spacy
from tqdm import tqdm


def normalize_text(text: str) -> str:
    """
    normalize_text returns normalized text.

    :param text: Text to process.
    :returns: Normalized string.

    Example usage:
        normalize_text(text='Some text here')
    """
    tm1 = re.sub('<pre>.*?</pre>', '', text, flags=re.DOTALL)
    tm2 = re.sub('<code>.*?</code>', '', tm1, flags=re.DOTALL)
    tm3 = re.sub('<[^>]+>©', '', tm2, flags=re.DOTALL)
    return tm3.replace("\n", "")


def cleanup_text(text: str, nlp: spacy.Language, stopwords: List[str]) -> str:
    """
    cleanup_text returns lemmatized text without punctuation symbols and stopwords.

    :param text: Text to process.
    :param nlp: spacy.Language object for language of text to be processed.
    :param stopwords: List of stopwords for selected language to be removed from processed text.
    :returns: Cleaned string.

    Example usage:
        cleanup_text(
            text = "Some text",
            nlp = spacy.load('en_core_web_lg'),
            stopwords = nltk.corpus.stopwords.words('english'))
    """
    punctuations = '!"#$%&\'()*+,-/:;<=>?@[\\]^_`{|}~©'
    doc = nlp(text, disable=['parser', 'ner'])
    tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
    tokens = [tok for tok in tokens if tok not in stopwords and tok not in punctuations]
    tokens = ' '.join(tokens)
    return tokens


def preprocess_text(
        text: str,
        nlp: spacy.Language = spacy.load('en_core_web_lg'),
        stopwords: List[str] = nltk.corpus.stopwords.words('english')) -> str:
    """
    preprocess_text returns normalized and cleaned text.

    :param text: Text to process.
    :param nlp: spacy.Language object for language of text to be processed.
    :param stopwords: List of stopwords for selected language to be removed from processed text.
    :returns: Cleaned string.

    Example usage:
        preprocess_text(
            text = "Some text",
            nlp = spacy.load('en_core_web_lg'),
            stopwords = nltk.corpus.stopwords.words('english'))
    """
    text = normalize_text(text)
    text = cleanup_text(text, nlp, stopwords)
    return text


def preprocess_texts(data: pd.DataFrame, column: str, preprocessed_column: str) -> pd.DataFrame:
    """
    preprocess_text returns a Pandas DataFrame with new column containing
    normalized and cleaned text from specified column values.

    :param data: A Pandas DataFrame object with text to process.
    :param column: Name of column with string values to be processed.
    :param preprocessed_column: Name of column to save processed text values in returned DataFrame object.
    :returns: A Pandas DataFrame object containing column with processed text values.

    Example usage:
        preprocess_texts(
            data = data,
            column = 'description',
            preprocessed_column = 'description_preprocessed')
    """
    data_copy = data.copy(deep=True)

    nlp = spacy.load('en_core_web_lg')
    stopwords = nltk.corpus.stopwords.words('english')

    tqdm.pandas()
    data_copy[preprocessed_column] = data_copy[column].progress_apply(lambda x: preprocess_text(x, nlp, stopwords))
    return data_copy
