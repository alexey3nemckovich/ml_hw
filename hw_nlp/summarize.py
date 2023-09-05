if __name__ == '__main__':
    import argparse
    from constants import *
    from logger import logger
    from preprocess_utils import preprocess_text
    from summarize_utils import summarize_text

    parser = argparse.ArgumentParser(description="This script performs wine description summarizing.")

    parser.add_argument("--description",
                        default="Oak and earth intermingle around robust aromas of wet "
                                "forest floor in this vineyard-designated "
                                "Pinot that hails from a high-elevation site. Small in production, it offers intense, "
                                "full-bodied raspberry and blackberry steeped in smoky spice and smooth texture.",
                        type=str,
                        help="Description to get summary for")

    parser.add_argument("--max_length",
                        default=DEFAULT_SUMMARY_MAX_LENGTH,
                        type=int,
                        help="Summary max length")

    args = parser.parse_args()

    DESCRIPTION = args.description
    SUMMARY_MAX_LENGTH = args.max_length

    logger.info(
        f"Summarizing wine description texts with parameters:\n"
        f"  description: '{DESCRIPTION}'\n"
        f"  max_length: {SUMMARY_MAX_LENGTH}")

    preprocessed_desc = preprocess_text(DESCRIPTION)

    logger.info(f"Preprocessed wine description:\n"
                f"  {preprocessed_desc}")

    summary = summarize_text(preprocessed_desc, SUMMARY_MAX_LENGTH)

    logger.info(f"Wine description summary:\n"
                f"  {summary}")
