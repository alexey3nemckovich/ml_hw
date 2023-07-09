# read

from utils import read_data
from utils import print_basic_statistics


data = read_data('./input/train.csv')

print_basic_statistics(data, 'mutation')

# check missings

from utils import show_data_missings

show_data_missings(data)

# base data preprocessing for EDA

from utils import encode_string_values
from sklearn.preprocessing import StandardScaler
import pandas as pd


std_scaler = StandardScaler()

eda_preprocessed_data = data.drop('ID', axis=1)
eda_preprocessed_data = encode_string_values(eda_preprocessed_data)
eda_preprocessed_data = pd.DataFrame(data=std_scaler.fit_transform(eda_preprocessed_data), columns=eda_preprocessed_data.columns)

# features correlation

from utils import plot_corr_matrix


plot_corr_matrix(eda_preprocessed_data, figsize=(30, 10))

# Top correlations:
# - M <> L
# - K <> U
# - R <> U
# - R <> K

# No big correlations between target and features

### Features distributions

from utils import pair_plot


pair_plot(eda_preprocessed_data, target='mutation')

# Base preprocessing

def drop_extra_features(data):
    return data.drop(['ID'], axis=1)

def get_features(data):
    return data.drop('mutation', axis=1)

from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import encode_string_values


preprocessing_steps = [
    ('select_features', FunctionTransformer(get_features)),
    ('drop_extra_features', FunctionTransformer(drop_extra_features)),
    ('label_encode', FunctionTransformer(encode_string_values)),
    ('scaler', StandardScaler())
]

preprocess_pipeline = Pipeline(preprocessing_steps)

X = preprocess_pipeline.fit_transform(data)

y = data.mutation

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

## PCA

from sklearn.decomposition import KernelPCA


kernel_pca = KernelPCA(n_components=X_train.shape[1], kernel='rbf', gamma=None, n_jobs=-1, random_state=42)

train_index = range(0,len(X_train))

X_train_PCA = kernel_pca.fit_transform(X_train)
X_train_PCA = kernel_pca.DataFrame(data=X_train_PCA, index=train_index)

importanceOfPrincipalComponents = pd.DataFrame(data=kernel_pca.explained_variance_ratio_)
importanceOfPrincipalComponents = importanceOfPrincipalComponents.T

print('Variance Captured by First 10 Principal Components: ',  importanceOfPrincipalComponents.loc[:,0:5].sum(axis=1).values)
print('Variance Captured by First 20 Principal Components: ',  importanceOfPrincipalComponents.loc[:,0:10].sum(axis=1).values)
print('Variance Captured by First 50 Principal Components: ',  importanceOfPrincipalComponents.loc[:,0:15].sum(axis=1).values)
print('Variance Captured by First 100 Principal Components: ', importanceOfPrincipalComponents.loc[:,0:20].sum(axis=1).values)
print('Variance Captured by First 200 Principal Components: ', importanceOfPrincipalComponents.loc[:,0:25].sum(axis=1).values)
print('Variance Captured by First 300 Principal Components: ', importanceOfPrincipalComponents.loc[:,0:30].sum(axis=1).values)

# CNN

from imblearn.under_sampling import CondensedNearestNeighbour

cnn = CondensedNearestNeighbour(n_jobs=-1)

X_train_resampled, y_train_resampled = cnn.fit_resample(X_train, y_train)
