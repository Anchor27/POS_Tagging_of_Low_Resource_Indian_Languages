import pandas as pd


''' DATA LOADING AAND PREPROCESSING '''
# Reading data in chunks
def read_csv_in_chunks(url, chunk_size=1000):
    reader = pd.read_csv(url, chunksize=chunk_size)
    for chunk in reader:
        yield chunk

data_chunks = []
for chunk in read_csv_in_chunks('/content/pos_bh.csv'):
    data_chunks.append(chunk)
bh_data = pd.concat(data_chunks, ignore_index=True)


combined_data=bh_data


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy import sparse


combined_data['tokens'] = combined_data['tokens'].fillna('')

# Converting the linguistic features to a sparse matrix
combined_data['text_length'] = combined_data['tokens'].apply(len)
combined_data['num_words'] = combined_data['tokens'].apply(lambda x: len(x.split()))
linguistic_features = combined_data[['text_length', 'num_words']]

combined_data = combined_data.sample(frac=1, random_state=42).reset_index(drop=True)

# test-train split
X_train_text, X_test_text, y_train, y_test = train_test_split(combined_data['tokens'], combined_data['pos_tags'], test_size=0.2, random_state=42)




''' FEATURE EXTRACTION '''

# Character n-gram features
char_vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), lowercase=True)
char_ngram_features = char_vectorizer.fit_transform(X_train_text)


# Converting char_ngram_features to sparse matrices
char_ngram_features = sparse.csr_matrix(char_ngram_features)


# Releasing memoory
del X_train_text

# Converting char_ngram_features to sparse matrices
linguistic_features = sparse.csr_matrix(linguistic_features)

# Ensuring linguistic_features and combined_features have compatible dimensions
if linguistic_features.shape[0] != char_ngram_features.shape[0]:
    linguistic_features = linguistic_features[:char_ngram_features.shape[0], :]

combined_features = sparse.hstack((char_ngram_features, linguistic_features)).tocsr()



''' MODEL TRAINING '''

#The Model
model = RandomForestClassifier(n_estimators=100, random_state=42)

#Training the Model
model.fit(combined_features, y_train)




''' PERFORMANCE ANALYSIS AN REPORTS '''

# Making predictions on the test data
X_test_char_ngram = char_vectorizer.transform(X_test_text)
X_test_linguistic = sparse.csr_matrix(linguistic_features)

# Ensuring test data dimensions are compatible
if X_test_linguistic.shape[0] != X_test_char_ngram.shape[0]:
    X_test_linguistic = X_test_linguistic[:X_test_char_ngram.shape[0], :]

X_test_combined = sparse.hstack((X_test_char_ngram, X_test_linguistic)).tocsr()
y_pred = model.predict(X_test_combined)

# Evaluating the model (tag - wise)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)


X_test_char_ngram = char_vectorizer.transform(X_test_text)

# Ensuring test data dimensions are compatible
if X_test_linguistic.shape[0] != X_test_char_ngram.shape[0]:
    X_test_linguistic = X_test_linguistic[:X_test_char_ngram.shape[0], :]

X_test_combined = sparse.hstack((X_test_char_ngram, X_test_linguistic)).tocsr()
y_pred = model.predict(X_test_combined)

# Calculating the overall metrics
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Extracting overall metrics
overall_accuracy = accuracy
overall_precision = report['macro avg']['precision']
overall_recall = report['macro avg']['recall']
overall_f1_score = report['macro avg']['f1-score']

# Printing the overall metrics
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Overall Precision: {overall_precision}")
print(f"Overall Recall: {overall_recall}")
print(f"Overall F1 Score: {overall_f1_score}")
