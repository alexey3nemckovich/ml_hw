if __name__ == '__main__':
    import argparse
    from summarize_utils import summarize_texts
    from file_utils import read_csv, save_int_to_file, file_exists
    from constants import *
    from logger import logger
    import sys

    parser = argparse.ArgumentParser(
        description="This script performs wines descriptions texts summarizing "
                    f"and saves data with new column '${SUMMARY_COLUMN}' to the 'target_file_path'."
    )

    parser.add_argument("--max_length",
                        default=DEFAULT_SUMMARY_MAX_LENGTH,
                        type=int,
                        help="Summary max length")

    parser.add_argument("--preprocessed_data_file_path",
                        default=DEFAULT_PREPROCESSED_DATA_FILE_PATH,
                        type=str,
                        help=f"Path to a data file with preprocessed wine descriptions "
                             f"(default: '{DEFAULT_PREPROCESSED_DATA_FILE_PATH}', type: str)")

    parser.add_argument("--target_file_path",
                        default=DEFAULT_SUMMARIZED_DATA_FILE_PATH,
                        type=str,
                        help=f"Path to a target data file to save data with descriptions summaries "
                             f"(default: '{DEFAULT_SUMMARIZED_DATA_FILE_PATH}', type: str)")

    args = parser.parse_args()

    SUMMARY_MAX_LENGTH = args.max_length
    PREPROCESSED_DATA_FILE_PATH = args.preprocessed_data_file_path
    TARGET_FILE_PATH = args.target_file_path

    if not file_exists(PREPROCESSED_DATA_FILE_PATH):
        logger.error("Invalid data file path!")
        sys.exit(1)

    logger.info(
        f"Summarizing wine reviews descriptions texts with parameters:\n"
        f"  max_length: {SUMMARY_MAX_LENGTH}\n"
        f"  preprocessed_data_file_path: '{PREPROCESSED_DATA_FILE_PATH}'\n"
        f"  target_file_path: '{TARGET_FILE_PATH}'")

    logger.info("Reading data file...")
    reviews = read_csv(PREPROCESSED_DATA_FILE_PATH)

    logger.info("Summarizing wines descriptions...")
    reviews = summarize_texts(reviews, DESCRIPTION_CLEANED_COLUMN, SUMMARY_COLUMN, SUMMARY_MAX_LENGTH)

    logger.info("Saving processed data to the target file...")
    reviews.to_csv(TARGET_FILE_PATH)

    logger.info("Saving used summarizing settings to the config file...")
    save_int_to_file(SUMMARY_CONFIG_FILE_PATH, SUMMARY_MAX_LENGTH)

    logger.info("Descriptions were successfully summarized!")
