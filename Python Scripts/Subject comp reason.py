import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
columns = ['SUBJECT_RACE', 'REASON_FOR_FORCE']

# Label-Encoding
encoder = LabelEncoder()
for column in columns:
    df[column] = encoder.fit_transform(df[column])
    # Translation num - cat
    print(f"Assignment of the numerical values to the categorical values in colum '{column}':")
    for i, category in enumerate(encoder.classes_):
        print(f"{i}: {category}")
    print()

# K-means Clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(df[columns])

# Add Clusterlabels
df['cluster_label'] = kmeans.labels_

# Dotplot
fig, ax = plt.subplots()

for label in df['cluster_label'].unique():
    cluster_data = df[df['cluster_label'] == label]
    x = cluster_data[columns[0]]
    y = cluster_data[columns[1]]
    ax.scatter(x, y, label=f'Cluster {label}')

ax.set_xlabel(columns[0])
ax.set_ylabel(columns[1])
ax.legend()

plt.show()