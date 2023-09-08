from typing import List

import gensim.models.doc2vec
import pandas as pd


def read_vectorized_data(filepath: str) -> pd.DataFrame:
    """
    read_data reads and returns Pandas DataFrame object from file with vectorized summaries.

    :param filepath: A path to a file with data.
    :returns: A Pandas DataFrame object.

    Example usage:
        read_vectorized_data(filepath='./data')
    """
    reviews_data = read_csv(filepath)
    reviews_data[VECTOR_COLUMN] = reviews_data[VECTOR_COLUMN].apply(str_to_ndarray)
    return reviews_data


def get_n_similar_wines(
        wine_summary: str,
        data: pd.DataFrame,
        doc2vec_model: gensim.models.doc2vec.Doc2Vec,
        learning_rate: float,
        vec_column: str,
        columns: List[str],
        n: int) -> pd.DataFrame:
    """
    get_n_similar_wines returns Pandas DataFrame object with information about 'n' most similar
    wines according to their vectorized summary representation.

    :param wine_summary: Wine summary.
    :param data: A Pandas DataFrame object with vectorized wines summaries representations.
    :param doc2vec_model: Trained Gensim Doc2Vec model object instance.
    :param learning_rate: Learning rate used to get wine summary vectorized representation.
    :param vec_column: Name of vectorized wines summaries representations in Pandas DataFrame object.
    :param columns: Columns with info to get for most similar wines.
    :param n: Count of similar wines to return
    :returns: Pandas DataFrame object.

    Example usage:
    get_n_similar_wines(
        wine_summary=summary,
        data=reviews,
        doc2vec_model=loaded_model,
        learning_rate=LEARNING_RATE,
        vec_column=VECTOR_COLUMN,
        columns=[TITLE_COLUMN, VARIETY_COLUMN, POINTS_COLUMN, PRICE_COLUMN, SIMILARITY_COLUMN],
        n=N_RECOMMENDS)
    """
    data_copy = data.copy(deep=True)

    vec = doc2vec_model.infer_vector(wine_summary.split(), alpha=learning_rate)

    data_copy[SIMILARITY_COLUMN] = data_copy[vec_column].apply(lambda x: cosine_similarity([vec], [x.tolist()])[0][0])

    return data_copy.sort_values(by=SIMILARITY_COLUMN, ascending=False)[columns][:n]


if __name__ == '__main__':
    import argparse
    from sklearn.metrics.pairwise import cosine_similarity
    from gensim.models.doc2vec import *
    from file_utils import read_csv, read_int_from_file, file_exists
    from constants import *
    from utils import str_to_ndarray
    from summarize_utils import summarize_text
    from preprocess_utils import preprocess_text
    import sys
    from logger import logger

    parser = argparse.ArgumentParser(
        description="This script gives 'n_recommends' for similar wines based on 'description' that was provided."
    )

    parser.add_argument("--description",
                        default="Much like the regular bottling from 2012, this comes across as rather rough and "
                                "tannic, with rustic, earthy, herbal characteristics. Nonetheless, if you think of it "
                                "as a pleasantly unfussy country wine, it's a good companion to a hearty winter stew.",
                        type=str,
                        help="Description to search similar wines for")

    parser.add_argument("--learning_rate",
                        default=DEFAULT_LEARNING_RATE,
                        type=float,
                        help="Learning rate to use for model")

    parser.add_argument("--vectorized_data_file_path",
                        default=DEFAULT_VECTORIZED_DATA_FILE_PATH,
                        type=str,
                        help=f"Path to a data file with vectorized wine summaries "
                             f"(default: '{DEFAULT_VECTORIZED_DATA_FILE_PATH}', type: str)")

    parser.add_argument("--doc2vec_model_file_path",
                        default=DEFAULT_DOC2VEC_MODEL_FILE_PATH,
                        type=str,
                        help=f"Path to a file with pretrained Doc2Vec model "
                             f"(default: '{DEFAULT_DOC2VEC_MODEL_FILE_PATH}', type: str)")

    parser.add_argument("--n_recommends",
                        default=DEFAULT_N_RECOMMENDS,
                        type=int,
                        help="Count of recommendations to give")

    args = parser.parse_args()

    DESCRIPTION = args.description
    LEARNING_RATE = args.learning_rate
    VECTORIZED_DATA_FILE_PATH = args.vectorized_data_file_path
    DOC2VEC_MODEL_FILE_PATH = args.doc2vec_model_file_path
    N_RECOMMENDS = args.n_recommends

    if not file_exists(VECTORIZED_DATA_FILE_PATH):
        logger.error("Invalid vectorized data file path! "
                     "If you haven't trained Doc2Vec model yet use 'train.py' script.")
        sys.exit(1)

    if not file_exists(DOC2VEC_MODEL_FILE_PATH):
        logger.error("Invalid Doc2Vec model file path!"
                     "If you haven't trained Doc2Vec model yet use 'train.py' script.")
        sys.exit(1)

    if not file_exists(SUMMARY_CONFIG_FILE_PATH):
        logger.error("Summary config file not found!"
                     "If you haven't summarized wines descriptions yet use 'summarize_texts.py' "
                     "and 'train.py' before using 'recommend.py' script.")
        sys.exit(1)

    reviews = read_vectorized_data(VECTORIZED_DATA_FILE_PATH)
    summary_max_length = read_int_from_file(SUMMARY_CONFIG_FILE_PATH)

    logger.info(
        f"Recommending similar wines with parameters:\n"
        f"  description: '{DESCRIPTION}'\n"
        f"  learning_rate: {LEARNING_RATE}\n"
        f"  vectorized_data_file_path: '{VECTORIZED_DATA_FILE_PATH}'\n"
        f"  doc2vec_model_file_path: '{DOC2VEC_MODEL_FILE_PATH}'\n"
        f"  n_recommends: {N_RECOMMENDS}")

    logger.info(f"Summarizing wine description...")
    summary = summarize_text(preprocess_text(DESCRIPTION), summary_max_length)

    logger.info(f"Wine description summary:\n"
                f"  {summary}")

    logger.info(f"Loading Doc2Vec model from file...")
    loaded_model = Doc2Vec.load(DOC2VEC_MODEL_FILE_PATH)

    logger.info(f"Searching for similar wines...")
    similar_wines = get_n_similar_wines(
        wine_summary=summary,
        data=reviews,
        doc2vec_model=loaded_model,
        learning_rate=LEARNING_RATE,
        vec_column=VECTOR_COLUMN,
        columns=[TITLE_COLUMN, VARIETY_COLUMN, POINTS_COLUMN, PRICE_COLUMN, SIMILARITY_COLUMN],
        n=N_RECOMMENDS)

    logger.info(f"Similar wines based on description provided: \n"
                f"{similar_wines}")
