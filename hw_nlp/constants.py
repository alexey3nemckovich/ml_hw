# file paths
DEFAULT_DATA_FILE_PATH = './wine_reviews.csv'
DEFAULT_PREPROCESSED_DATA_FILE_PATH = './wine_reviews_preprocessed.csv'
DEFAULT_SUMMARIZED_DATA_FILE_PATH = './wine_reviews_summarized.csv'
DEFAULT_VECTORIZED_DATA_FILE_PATH = './wine_reviews_vectorized.csv'
DEFAULT_DOC2VEC_MODEL_FILE_PATH = './doc2vec_model.pkl'
SUMMARY_CONFIG_FILE_PATH = './summary_config.txt'
NLP_LOG = './nlp.log'

# columns
DESCRIPTION_COLUMN = 'description'
DESCRIPTION_CLEANED_COLUMN = 'description_cleaned'
SUMMARY_COLUMN = 'summary'
VECTOR_COLUMN = 'vec'
SIMILARITY_COLUMN = 'similarity'
TITLE_COLUMN = 'title'
VARIETY_COLUMN = 'variety'
POINTS_COLUMN = 'points'
PRICE_COLUMN = 'price'
PRICE_AVG_COLUMN = 'price_avg'

# summarizing params
DEFAULT_SUMMARY_MAX_LENGTH = 50

# training params
DEFAULT_VECTOR_SIZE = 100
DEFAULT_WINDOWS_SIZE = 5
DEFAULT_MIN_COUNT = 1
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_LEARNING_RATE = 0.025
DEFAULT_MIN_LEARNING_RATE = 0.01

DEFAULT_N_RECOMMENDS = 5

NLP_LOGGER_NAME = 'default_nlp_logger'
LOGGER_FORMAT = '[%(asctime)s][%(levelname)s] %(message)s'
