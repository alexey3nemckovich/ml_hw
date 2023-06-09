import pandas as pd
import plotly.express as px
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from plotly.offline import init_notebook_mode


def read_data(file_path):
    return pd.read_csv(file_path)


def print_target_distribution(data, target_name, notebook_mode=False):
    if notebook_mode:
        init_notebook_mode(connected=True)

    mutation_counts = data[target_name].value_counts()
    fig = px.bar(mutation_counts, x=mutation_counts.index, y=mutation_counts.values, title='Class distribution')
    fig.show()


def print_basic_statistics(data, target_name):
    pd.set_option('display.width', 1000)
    print(f'Shape = {data.shape}\n')
    print(f'Columns = {data.columns.values}\n')
    print(f'First 15 samples:\n{data.head(15)}\n')
    print(f'Basic statistics:\n{data.describe().T}\n')
    print_target_distribution(data, target_name)


def show_data_missings(data):
    msno.matrix(data)
    plt.figure(figsize = (15,9))
    plt.show()


def label_encode_string_values(data):
    data_encoded = data.copy()
    encoder = LabelEncoder()

    for column in data.columns:
        if data_encoded[column].dtype == 'object':
            data_encoded[column] = encoder.fit_transform(data[column])

    return data_encoded


def get_string_columns(data):
    string_columns = []

    for column in data.columns:
        if data[column].dtype == 'object':
            string_columns.append(column)

    return string_columns


def get_numerical_columns(data):
    numerical_columns = []

    for column in data.columns:
        if data[column].dtype != 'object':
            numerical_columns.append(column)

    return numerical_columns


def onehot_encode_string_values(data):
    data_encoded = data.copy()
    string_columns = get_string_columns(data)

    for column in string_columns:
        df_encoded_column = pd.get_dummies(data[column], drop_first=True, prefix=column).astype(int)

        data_encoded = pd.concat([data_encoded.drop(column, axis=1), df_encoded_column], axis=1)            

    return data_encoded


def std_scale_numerical_values(data):
    data_scaled = data.copy()
    scaler_class = StandardScaler
    numerical_columns = get_numerical_columns(data)

    for column in numerical_columns:
        data_scaled[column] = scaler_class().fit_transform(data_scaled[column])
    
    return data_scaled


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)

    conf_matrix = pd.DataFrame(data = cm, columns = ['Predicted:0','Predicted:1'], index=['Actual:0','Actual:1'])
    plt.figure(figsize = (5,5))
    sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu", cbar=False);


def plot_roc_curve(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    
    plt.plot(fpr, tpr, label='ROC curve ')
    plt.plot([0, 1], [0, 1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC: {:.3f}'.format(auc_score))
    plt.show()


def plot_corr_matrix(data, figsize):
    plt.figure(figsize=figsize)
    sns.heatmap(data.corr(), annot=True);


def pair_plot(data, target):
    sns.pairplot(data, hue=target);


def get_basic_quality_metrics(y_true, y_pred):
    roc_auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return roc_auc, accuracy, precision, recall, f1


def print_basic_quality_metrics(y_true, y_pred):
    roc_auc, accuracy, precision, recall, f1 = get_basic_quality_metrics(y_true, y_pred)
    
    print("ROC AUC:   {:.3f}\nAccuracy:  {:.3f}\nPrecision: {:.3f}\nRecall:    {:.3f}\nF1-score:  {:.3f}".format(
        roc_auc, accuracy, precision, recall, f1
    ))


def basic_model_test(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print_basic_quality_metrics(y_test, y_pred)


def ndarray_to_dataframe(arr):
    arr_index = range(0, len(arr))

    return pd.DataFrame(data=arr, index=arr_index)


def ndarray_to_csv(arr, path):
    dataframe = ndarray_to_dataframe(arr)

    dataframe.to_csv(path)


def plot_explained_variance_ratio(explained_variance_ratio, algo_name):
    plt.bar(range(0, len(explained_variance_ratio)), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Explained Variance Ratio of {algo_name}')
    plt.show()


def plot_cumulative_explained_variance_ratio(explained_variance_ratio, algo_name, threshold):
    explained_variance_ratio = explained_variance_ratio.copy()
    enough_features = -1

    for i in range(1, len(explained_variance_ratio)):
        explained_variance_ratio[i] += explained_variance_ratio[i-1]
        if (enough_features == -1):
            if (explained_variance_ratio[i] >= threshold):
                enough_features = i + 1
                print(f'First {enough_features} features explain enough variance for threshold {threshold}: {explained_variance_ratio[i]}')
     
    plt.bar(range(0, len(explained_variance_ratio)), explained_variance_ratio)
    plt.xlabel('Principal Component')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(f'Cumulative Explained Variance Ratio of {algo_name}')
    plt.axhline(y=threshold, color='red', linestyle='--')

    plt.show()

    return enough_features
