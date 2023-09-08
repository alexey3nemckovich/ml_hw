import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
tokenizer = AutoTokenizer.from_pretrained('t5-base')


def summarize_text(text: str, max_length: int) -> str:
    """
    summarize_text returns text summary.

    :param text: Text to summarize.
    :param max_length: Max length of summary.
    :returns: Text summary

    Example usage:
    summarize_text(text='Some text',
        max_length=70)
    """
    tokens_input = tokenizer.encode("summarize: " + text,
                                    return_tensors='pt',
                                    max_length=512,
                                    truncation=True)

    summary_ids = model.generate(tokens_input, max_length=max_length, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_texts(data: pd.DataFrame, column: str, summary_column: str, summary_max_length: int) -> pd.DataFrame:
    """
    summarize_texts returns Pandas DataFrame object with new column
    containing summaries for texts in specified dataframe column.

    :param data: Pandas DataFrame object with data to process.
    :param column: Name of column with text values to summarize.
    :param summary_column: Name of column to be created for summaries values in returned DataFrame object.
    :param summary_max_length: Max length of summary.
    :returns: Pandas DataFrame object with summaries values.

    Example usage:
    summarize_texts(
        data=reviews,
        column=DESCRIPTION_CLEANED_COLUMN,
        summary_column=SUMMARY_COLUMN,
        summary_max_length=SUMMARY_MAX_LENGTH)
    """
    data_copy = data.copy(deep=True)

    tqdm.pandas()
    data_copy[summary_column] = data_copy[column].progress_apply(lambda x: summarize_text(x, summary_max_length))
    return data_copy
