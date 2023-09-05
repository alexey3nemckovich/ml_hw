if __name__ == '__main__':
    import argparse
    import sys
    from preprocess_utils import preprocess_texts
    from file_utils import read_csv, file_exists
    from constants import *
    from logger import logger

    parser = argparse.ArgumentParser(
        description="This script performs basic wines descriptions texts preprocessing "
                    f"and saves data with new column '${DESCRIPTION_CLEANED_COLUMN}' to the 'target_file_path'."
    )

    parser.add_argument("--data_file_path",
                        default=DEFAULT_DATA_FILE_PATH,
                        type=str,
                        help=f"Path to a source data file (default: '{DEFAULT_DATA_FILE_PATH}', type: str)")

    parser.add_argument("--target_file_path",
                        default=DEFAULT_PREPROCESSED_DATA_FILE_PATH,
                        type=str,
                        help=f"Path to a target data file to save data with preprocessed descriptions (default: '{DEFAULT_PREPROCESSED_DATA_FILE_PATH}', type: str)")

    args = parser.parse_args()

    DATA_FILE_PATH = args.data_file_path
    TARGET_FILE_PATH = args.target_file_path

    if not file_exists(DATA_FILE_PATH):
        logger.error("Invalid data file path!")
        sys.exit(1)

    logger.info(
        f"Preprocessing wine reviews descriptions texts with parameters:\n"
        f"  data_file_path: '{DATA_FILE_PATH}'\n"
        f"  target_file_path: '{TARGET_FILE_PATH}'")

    logger.info("Reading data file...")
    reviews = read_csv(DATA_FILE_PATH)

    logger.info("Preprocessing wines descriptions...")
    reviews = preprocess_texts(reviews, DESCRIPTION_COLUMN, DESCRIPTION_CLEANED_COLUMN)

    logger.info("Saving processed data to the target file...")
    reviews.to_csv(TARGET_FILE_PATH)

    logger.info("Data was successfully preprocessed!")
