import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder
from kmodes.kmodes import KModes

# Loading process

# Path to both chosen files
file1 = r'E:\Dropbox\Studium\Unsupervised Maschine Learning\Data Task 2\archive\Chosen\23-00089_UOF-P.csv'
file2 = r'E:\Dropbox\Studium\Unsupervised Maschine Learning\Data Task 2\archive\Chosen\37-00027_UOF-P_2014-2016_prepped.csv'

# Load files as data frames:
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine data frames
df = pd.concat([df1, df2])
# Choose columns
columns = ['SUBJECT_RACE', 'OFFICER_RACE']

# Label-Encoding
encoder = LabelEncoder()
for column in columns:
    df[column] = encoder.fit_transform(df[column])

# Prepare Calculation
X = df[columns]

# Elbow method:
sse = []
for k in range(2, 11):
    kmodes = KModes(n_clusters=k)
    kmodes.fit(X)
    sse.append(kmodes.cost_)

# Plot Elbow method:
plt.plot(range(2, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow method: Sum of squared deviations (SSE)')
plt.show()

# Silhouette coefficient:
silhouette_scores = []
for k in range(2, 11):
    kmodes = KModes(n_clusters=k)
    kmodes.fit(X)
    labels = kmodes.labels_
    silhouette_scores.append(silhouette_score(X, labels))

# Plot Silhouette coefficient
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette coefficient')
plt.title('Silhouette coefficient')
plt.show()
