import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
tokenizer = AutoTokenizer.from_pretrained('t5-base')


def summarize_text(text: str, max_length: int):
    tokens_input = tokenizer.encode("summarize: " + text,
                                    return_tensors='pt',
                                    max_length=512,
                                    truncation=True)

    summary_ids = model.generate(tokens_input, max_length=max_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def summarize_texts(data: pd.DataFrame, column: str, summary_column: str, summary_max_length: int):
    data_copy = data.copy(deep=True)

    tqdm.pandas()
    data_copy[summary_column] = data_copy[column].progress_apply(lambda x: summarize_text(x, summary_max_length))
    return data_copy
