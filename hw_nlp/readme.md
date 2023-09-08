
# ML project to recommend similar wines based on their descriptions

We have dataset of 22387 wines. This project helps to get some most similar of them according to the provided description.

## Getting summary:

To get summary for single wine description you can use script **'./summarize.py'**

### Example of execution:
````
  (venv) alex@alex-pc:~/projects/ml/ml_hw/hw_nlp$ python ./summarize.py
````
[2023-09-08 14:02:30][INFO] Summarizing wine description texts with parameters:
  description: 'Oak and earth intermingle around robust aromas of wet forest floor in this vineyard-designated Pinot that hails from a high-elevation site. Small in production, it offers intense, full-bodied raspberry and blackberry steeped in smoky spice and smooth texture.'
  max_length: 50
[2023-09-08 14:02:30][INFO] Preprocessed wine description:
  oak earth intermingle around robust aroma wet forest floor vineyard designate pinot hail high elevation site . small production offer intense full bodied raspberry blackberry steep smoky spice smooth texture .
[2023-09-08 14:02:31][INFO] Wine description summary:
  small production offer intense full bodied raspberry blackberry steep smoky spice smooth texture.
  
### Script usage information:
usage: summarize.py [-h] [--description DESCRIPTION] [--max_length MAX_LENGTH]

This script performs wine description summarizing.

options:
  -h, --help            show this help message and exit
  --description DESCRIPTION
                        Description to get summary for
  --max_length MAX_LENGTH
                        Summary max length

## Getting recommendation:
### Running steps:
   1. Preprocessing wines descriptions using **'preprocess_texts.py'** script. Script will save wines with their descriptions cleaned representation into separate CSV file (by default './wine_reviews_preprocessed.csv')
   2. Summarizing wines descriptions using **'summarize_texts.py'** script. Script will save wines with their descriptions summaries into separate CSV file (by default './wine_reviews_summarized.csv')
   3. Vectorizing wines summaries using **'train.py'**. Script will train Doc2Vec model to use it later to get vectorized representations for new summaries. Wines with their vectorized summaries representations will be saved into separate file (by default './wine_reviews_vectorized.csv'). Trained Doc2Vec model will also be saved into separate file (by default './doc2vec_model.pkl')
   4. After all steps mentioned above you can take recommendations for most similar wines using **'./recommend.py'** scipt. It will print out information about N most similar lines.

### Example of execution:
````
  (venv) alex@alex-pc:~/projects/ml/ml_hw/hw_nlp$ python ./recommend.py
````
[2023-09-08 15:04:52][INFO] Recommending similar wines with parameters:
  description: 'Much like the regular bottling from 2012, this comes across as rather rough and tannic, with rustic, earthy, herbal characteristics. Nonetheless, if you think of it as a pleasantly unfussy country wine, it's a good companion to a hearty winter stew.'
  learning_rate: 0.025
  vectorized_data_file_path: './wine_reviews_vectorized.csv'
  doc2vec_model_file_path: './doc2vec_model.pkl'
  n_recommends: 5
[2023-09-08 15:04:52][INFO] Summarizing wine description...
[2023-09-08 15:04:54][INFO] Wine description summary:
  much like regular bottling 2012 come across rather rough tannic rustic earthy herbal characteristic. nevertheless think pleasantly unfussy country wine good companion hearty winter stew.
[2023-09-08 15:04:54][INFO] Loading Doc2Vec model from file...
[2023-09-08 15:04:54][INFO] Searching for similar wines...
[2023-09-08 15:04:58][INFO] Similar wines based on description provided: 
                                                   title     variety  points  price_avg  similarity
0      Sweet Cheeks 2012 Vintner's Reserve Wild Child...  Pinot Noir      87       65.0    0.994930
3184   Block Nine 2014 Caiden's Vineyards Pinot Noir ...  Pinot Noir      82       14.0    0.526929
16191  Kokomo 2013 Pauline's Vineyard Merlot (Dry Cre...      Merlot      89       34.0    0.521924
7326            Hyatt 2000 Reserve Syrah (Yakima Valley)       Syrah      83       18.0    0.517578
13894  Amici 2012 Charles Heintz Vineyard Chardonnay ...  Chardonnay      84       65.0    0.506240

### Scripts usage information:
#### preprocess_text.py
usage: preprocess_texts.py [-h] [--data_file_path DATA_FILE_PATH] [--target_file_path TARGET_FILE_PATH]

This script performs basic wines descriptions texts preprocessing and saves data with new column '$description_cleaned' to the 'target_file_path'.

options:
  -h, --help            show this help message and exit
  --data_file_path DATA_FILE_PATH
                        Path to a source data file (default: './wine_reviews.csv', type: str)
  --target_file_path TARGET_FILE_PATH
                        Path to a target data file to save data with preprocessed descriptions (default: './wine_reviews_preprocessed.csv', type: str)

#### summarize_texts.py
usage: summarize_texts.py [-h] [--max_length MAX_LENGTH] [--preprocessed_data_file_path PREPROCESSED_DATA_FILE_PATH] [--target_file_path TARGET_FILE_PATH]

This script performs wines descriptions texts summarizing and saves data with new column '$summary' to the 'target_file_path'.

options:
  -h, --help            show this help message and exit
  --max_length MAX_LENGTH
                        Summary max length
  --preprocessed_data_file_path PREPROCESSED_DATA_FILE_PATH
                        Path to a data file with preprocessed wine descriptions (default: './wine_reviews_preprocessed.csv', type: str)
  --target_file_path TARGET_FILE_PATH
                        Path to a target data file to save data with descriptions summaries (default: './wine_reviews_summarized.csv', type: str)

#### train.py
usage: train.py [-h] [--vector_size VECTOR_SIZE] [--window_size WINDOW_SIZE] [--min_count MIN_COUNT] [--num_epochs NUM_EPOCHS] [--learning_rate LEARNING_RATE] [--min_learning_rate MIN_LEARNING_RATE]
                [--summarized_data_file_path SUMMARIZED_DATA_FILE_PATH] [--target_file_path TARGET_FILE_PATH] [--doc2vec_model_file_path DOC2VEC_MODEL_FILE_PATH]

This script performs Doc2Vec model training and saves data with new column 'vec' to the 'target_file_path' and trained Doc2Vec model to the 'doc2vec_model_file_path'.

options:
  -h, --help            show this help message and exit
  --vector_size VECTOR_SIZE
                        Vector size to use for summaries presentation
  --window_size WINDOW_SIZE
                        Windows size
  --min_count MIN_COUNT
                        Min count of word appearances
  --num_epochs NUM_EPOCHS
                        The number of epochs to train for
  --learning_rate LEARNING_RATE
                        Learning rate to use for model
  --min_learning_rate MIN_LEARNING_RATE
                        Min learning rate to use for model
  --summarized_data_file_path SUMMARIZED_DATA_FILE_PATH
                        Path to a data file with summarized wine descriptions (default: './wine_reviews_summarized.csv', type: str)
  --target_file_path TARGET_FILE_PATH
                        Path to a target data file to save data with vectorized wines summaries (default: './wine_reviews_vectorized.csv', type: str)
  --doc2vec_model_file_path DOC2VEC_MODEL_FILE_PATH
                        Path to a file to save trained Doc2Vec model (default: './doc2vec_model.pkl', type: str)

#### recommend.py
usage: recommend.py [-h] [--description DESCRIPTION] [--learning_rate LEARNING_RATE] [--vectorized_data_file_path VECTORIZED_DATA_FILE_PATH] [--doc2vec_model_file_path DOC2VEC_MODEL_FILE_PATH] [--n_recommends N_RECOMMENDS]

This script gives 'n_recommends' for similar wines based on 'description' that was provided.

options:
  -h, --help            show this help message and exit
  --description DESCRIPTION
                        Description to search similar wines for
  --learning_rate LEARNING_RATE
                        Learning rate to use for model
  --vectorized_data_file_path VECTORIZED_DATA_FILE_PATH
                        Path to a data file with vectorized wine summaries (default: './wine_reviews_vectorized.csv', type: str)
  --doc2vec_model_file_path DOC2VEC_MODEL_FILE_PATH
                        Path to a file with pretrained Doc2Vec model (default: './doc2vec_model.pkl', type: str)
  --n_recommends N_RECOMMENDS
                        Count of recommendations to give
