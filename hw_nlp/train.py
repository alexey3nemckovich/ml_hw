import pandas as pd


def build_tagged_documents(data: pd.DataFrame, text_column: str):
    splitted_texts = [text.split() for text in data[text_column]]
    idx = [i for i in range(len(data))]

    documents = []
    for i in range(len(data)):
        documents.append(TaggedDocument(splitted_texts[i], [idx[i]]))

    return documents


if __name__ == '__main__':
    import argparse
    from gensim.models.doc2vec import *
    from file_utils import read_csv, file_exists
    from progress_callback import ProgressCallback
    from logger import logger
    from constants import *
    import sys

    parser = argparse.ArgumentParser(
        description="This script performs Doc2Vec model training"
                    f" and saves data with new column '{VECTOR_COLUMN}' to the 'target_file_path'"
                    f" and trained Doc2Vec model to the 'doc2vec_model_file_path'."
    )

    parser.add_argument("--vector_size",
                        default=DEFAULT_VECTOR_SIZE,
                        type=int,
                        help="Vector size to use for summaries presentation")

    parser.add_argument("--window_size",
                        default=DEFAULT_WINDOWS_SIZE,
                        type=int,
                        help="Windows size")

    parser.add_argument("--min_count",
                        default=DEFAULT_MIN_COUNT,
                        type=int,
                        help="Min count of word appearances")

    parser.add_argument("--num_epochs",
                        default=DEFAULT_NUM_EPOCHS,
                        type=int,
                        help="The number of epochs to train for")

    parser.add_argument("--learning_rate",
                        default=DEFAULT_LEARNING_RATE,
                        type=float,
                        help="Learning rate to use for model")

    parser.add_argument("--min_learning_rate",
                        default=DEFAULT_MIN_LEARNING_RATE,
                        type=float,
                        help="Min learning rate to use for model")

    parser.add_argument("--summarized_data_file_path",
                        default=DEFAULT_SUMMARIZED_DATA_FILE_PATH,
                        type=str,
                        help=f"Path to a data file with summarized wine descriptions "
                             f"(default: '{DEFAULT_SUMMARIZED_DATA_FILE_PATH}', type: str)")

    parser.add_argument("--target_file_path",
                        default=DEFAULT_VECTORIZED_DATA_FILE_PATH,
                        type=str,
                        help=f"Path to a target data file to save data with vectorized wines summaries "
                             f"(default: '{DEFAULT_VECTORIZED_DATA_FILE_PATH}', type: str)")

    parser.add_argument("--doc2vec_model_file_path",
                        default=DEFAULT_DOC2VEC_MODEL_FILE_PATH,
                        type=str,
                        help=f"Path to a file to save trained Doc2Vec model "
                             f"(default: '{DEFAULT_DOC2VEC_MODEL_FILE_PATH}', type: str)")

    args = parser.parse_args()

    VECTOR_SIZE = args.vector_size
    WINDOW_SIZE = args.window_size
    MIN_COUNT = args.min_count
    NUM_EPOCHS = args.num_epochs
    LEARNING_RATE = args.learning_rate
    MIN_LEARNING_RATE = args.min_learning_rate
    SUMMARIZED_DATA_FILE_PATH = args.summarized_data_file_path
    TARGET_FILE_PATH = args.target_file_path
    MODEL_FILE_PATH = args.doc2vec_model_file_path

    if not file_exists(SUMMARIZED_DATA_FILE_PATH):
        logger.error("Invalid summarized data file path!"
                     "If you haven't summarized wines descriptions yet use 'summarize_texts.py'.")
        sys.exit(1)

    if not file_exists(SUMMARY_CONFIG_FILE_PATH):
        logger.error("Summary config file not found!"
                     "If you haven't summarized wines descriptions yet use 'summarize_texts.py'.")
        sys.exit(1)

    logger.info(
        f"Training Doc2Vec model with parameters:\n"
        f"  vector_size: {VECTOR_SIZE}\n"
        f"  window_size: {WINDOW_SIZE}\n"
        f"  min_count: {MIN_COUNT}\n"
        f"  num_epochs: {NUM_EPOCHS}\n"
        f"  learning_rate: {LEARNING_RATE}\n"
        f"  min_learning_rate: {MIN_LEARNING_RATE}\n"
        f"  summarized_data_file_path: '{SUMMARIZED_DATA_FILE_PATH}'\n"
        f"  target_file_path: '{TARGET_FILE_PATH}'\n"
        f"  doc2vec_model_file_path: '{MODEL_FILE_PATH}'")

    logger.info("Reading data file...")
    reviews = read_csv(SUMMARIZED_DATA_FILE_PATH)

    logger.info("Training Doc2Vec model...")
    docs = build_tagged_documents(reviews, SUMMARY_COLUMN)

    model = Doc2Vec(vector_size=VECTOR_SIZE,
                    window=WINDOW_SIZE,
                    min_count=MIN_COUNT,
                    workers=8,
                    alpha=LEARNING_RATE,
                    min_alpha=MIN_LEARNING_RATE,
                    dm=0)

    model.build_vocab(docs)
    model.train(docs, total_examples=len(docs), epochs=NUM_EPOCHS, callbacks=[ProgressCallback(logger)])

    reviews[VECTOR_COLUMN] = reviews.index.to_series().apply(lambda doc_id: model.dv[doc_id])

    logger.info("Saving model data to the file...")
    model.save(MODEL_FILE_PATH)

    logger.info("Saving processed data to the target file...")
    reviews.to_csv(TARGET_FILE_PATH)

    logger.info("Doc2Vec model was successfully trained and saved!")
