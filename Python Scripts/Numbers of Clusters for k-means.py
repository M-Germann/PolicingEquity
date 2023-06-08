import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

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
columns = ['SUBJECT_RACE', 'REASON_FOR_FORCE', 'OFFICER_RACE']

# Label-Encoding
encoder = LabelEncoder()
for column in columns:
    df[column] = encoder.fit_transform(df[column])

# Prepare Calculation
X = df[columns]

# Elbow method:
sse = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# Plot Elbow method:
plt.plot(range(2, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow method: Sum of squared deviations (SSE)')
plt.show()

# Silhouette coefficient:
silhouette_scores = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    silhouette_scores.append(silhouette_score(X, kmeans.labels_))

# Plot Silhouette coefficient
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette coefficient')
plt.title('Silhouette coefficient')
plt.show()