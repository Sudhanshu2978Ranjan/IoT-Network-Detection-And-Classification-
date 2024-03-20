import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
file_path = "S://Dataset//c1.csv"
original_df= pd.read_csv(file_path,delimiter = "|")
random_sample_df = original_df.sample(n=200000, random_state=42)

# Create a copy of the randomly sampled DataFrame to avoid modifying the original data.
df = random_sample_df.copy()
print(df.head)
# Replace hyphens ('-') with pandas' NA values for better handling of missing or undefined data.
df.replace('-', pd.NA, inplace=True)

print(df.head)
null_values = df.isnull().sum()
print(null_values)
null_values = df.isnull().sum()
print(null_values)
# Plotting null values using seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x=null_values.index, y=null_values)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Columns')
plt.ylabel('Number of Null Values')
plt.title('Null Values in Malware_Detect_Data.csv')
plt.tight_layout()
plt.show()
# Generate a heatmap of null values
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Heatmap of Null Values in c1.csv')
plt.show()
null_values = df.isnull().sum()

null_percentage = (null_values / len(df)) * 100

columns_with_null = null_percentage[null_percentage > 0]

print("Columns with Null Values (Percentage):")
print(columns_with_null)
# droping columns with null values
df.drop(['service','orig_bytes','resp_bytes','local_orig','local_resp','tunnel_parents'],axis=1,inplace=True)
#Generate a heatmap of null values
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Heatmap of Null Values in c1.csv')
plt.show()
label_encoder = preprocessing.LabelEncoder()
#label encode label
df['label']= label_encoder.fit_transform(df['label'])
df.head()
df['duration'] = pd.to_numeric(df['duration'])
selected_columns = ['duration', 'label']

# Create a DataFrame with only the selected columns
selected_df = df[selected_columns]

# Calculate the correlation matrix
correlation_matrix = selected_df.corr()

plt.figure(figsize=(10, 8))

# Plot a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=0.5)

plt.title('Correlation Matrix: Duration and Label')

plt.show()
df.drop(['duration'],axis=1,inplace=True)
df['history'].value_counts()
#removing null values
df.dropna(subset=['history'], inplace=True)
label_encoder = preprocessing.LabelEncoder()
df['history']= label_encoder.fit_transform(df['history'])
df['history'].unique()
df['detailed-label'].value_counts()
df.drop(df[df['detailed-label'] == 'C&C'].index, inplace = True)
df['detailed-label'].value_counts()
df['detailed-label'].fillna('n', inplace=True)
#onehot encode
onehot = pd.get_dummies(df['detailed-label'])
df = df.join(onehot)
df.head()
df.drop(['detailed-label'],axis=1,inplace=True)
df.head()
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cmap='viridis', cbar=False)
plt.xlabel('Columns')
plt.ylabel('Rows')
plt.title('Heatmap of Null Values c1.csv')
plt.show()
df['ts'] = pd.to_numeric(df['ts'])
df['ts'].mean()
# types of timestamp
df['uid'].value_counts()
df['uid']= label_encoder.fit_transform(df['uid'])
# types of timestamp
df['uid'].value_counts()
df['id.orig_h'].value_counts()
df['id.orig_h']= label_encoder.fit_transform(df['id.orig_h'])
# types of timestamp
df['id.orig_h'].value_counts()
df['id.resp_h'].value_counts()
df['id.resp_h']= label_encoder.fit_transform(df['id.resp_h'])
df['id.resp_p'].value_counts()
df['id.resp_h'].value_counts()
df['id.resp_p']= label_encoder.fit_transform(df['id.resp_p'])
df['id.resp_p'].value_counts()
df['id.orig_p'].value_counts()
df['id.orig_p']= label_encoder.fit_transform(df['id.orig_p'])
df['id.orig_p'].value_counts()
df['proto'].value_counts()
#one hot encode proto
onehot = pd.get_dummies(df['proto'])
df = df.join(onehot)
df.head()
df.drop(['proto'],axis=1,inplace=True)
df.head()
df['conn_state'].value_counts()
#label encode conn_state
df['conn_state']= label_encoder.fit_transform(df['conn_state'])
df['conn_state'].value_counts()
df['missed_bytes'].value_counts()
df.drop(['missed_bytes'],axis=1,inplace=True)
df['orig_pkts'].value_counts()
df['orig_pkts'] = pd.to_numeric(df['orig_pkts'])
df['orig_ip_bytes'] = pd.to_numeric(df['orig_ip_bytes'])
df['resp_pkts'] = pd.to_numeric(df['resp_pkts'])
df['resp_ip_bytes'] = pd.to_numeric(df['resp_ip_bytes'])
df['label'].value_counts()
df['label']= label_encoder.fit_transform(df['label'])
df.head()
from sklearn.model_selection import train_test_split

X = df.drop('label', axis=1)
y = df['label']

# Split the DataFrame into X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train.head()
y_train.value_counts()
from sklearn.preprocessing import Normalizer

scaler = Normalizer()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

# Assuming X_train_scaled, X_test_scaled, y_train, y_test are already defined

# Train SVM model
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = svm_model.predict(X_test_scaled)

# Calculate accuracy on test set
accuracy_test = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy_test)

# Get decision function values for plotting
dec_func_train = svm_model.decision_function(X_train_scaled)
dec_func_test = svm_model.decision_function(X_test_scaled)

# Plot training/testing curves
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.hist(dec_func_train, bins=50, alpha=0.7, color='blue', label='Train')
plt.hist(dec_func_test, bins=50, alpha=0.7, color='red', label='Test')
plt.title('Decision Function Values')
plt.xlabel('Decision Function Value')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(y_test)), y_test, color='black', label='True Values')
plt.scatter(range(len(y_pred)), y_pred, color='orange', marker='x', label='Predicted Values')
plt.title('True vs Predicted Values')
plt.xlabel('Sample Index')
plt.ylabel('Class')
plt.legend()

plt.tight_layout()
plt.show()
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb)

print(f"Accuracy (Naive Bayes): {accuracy_nb}")
print(f"Classification Report (Naive Bayes):\n{report_nb}")
from sklearn.model_selection import cross_val_score, KFold


# number of folds for cross-validation
k_folds = 5

kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

clf_nb =nb_model

# performing k-fold cross-validation
cross_val_results = cross_val_score(clf_nb, X_train_scaled, y_train, cv=kf, scoring='accuracy')

# results
print(f'Cross-validation results: {cross_val_results}')
print(f'Mean accuracy: {cross_val_results.mean()}')
from sklearn.metrics import accuracy_score

# Predict the labels for the training set
y_train_pred = nb_model.predict(X_train)

# Calculate the training accuracy
training_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", training_accuracy)
from sklearn.metrics import accuracy_score

# Calculate test accuracy
test_accuracy = accuracy_score(y_test, y_pred_nb)
print("Test Accuracy:", test_accuracy)
from sklearn.metrics import accuracy_score

# Calculate validation score
validation_score = accuracy_score(y_test, y_pred_nb)
print("Validation Score (Accuracy):", validation_score)

