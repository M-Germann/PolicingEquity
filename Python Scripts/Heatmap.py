import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder

# Dateipfade der CSV-Dateien
file1 = r'E:\Dropbox\Studium\Unsupervised Maschine Learning\Data Task 2\archive\Chosen\23-00089_UOF-P.csv'
file2 = r'E:\Dropbox\Studium\Unsupervised Maschine Learning\Data Task 2\archive\Chosen\37-00027_UOF-P_2014-2016_prepped.csv'

# CSV-Dateien in Dataframes laden
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Dataframes zusammenführen
df = pd.concat([df1, df2])

# Kategorische Spalten für die Clusteranalyse auswählen
columns = ['SUBJECT_RACE', 'REASON_FOR_FORCE', 'OFFICER_RACE']

# Label-Encoding der kategorischen Spalten
encoder = LabelEncoder()
for column in columns:
    df[column] = encoder.fit_transform(df[column])
    # Zuordnung der numerischen Werte zu den kategorischen Werten ausgeben
    print(f"Zuordnung der numerischen Werte zu den kategorischen Werten in Spalte '{column}':")
    for i, category in enumerate(encoder.classes_):
        print(f"{i}: {category}")
    print()

# K-means Clustering durchführen
kmeans = KMeans(n_clusters=3)  # Anzahl der Cluster anpassen
kmeans.fit(df[columns])

# Clusterlabels zu den Daten hinzufügen
df['cluster_label'] = kmeans.labels_

# Erstellen der Kreuztabelle für die Heatmap
heatmap_data = pd.crosstab(index=df['cluster_label'], columns=[df[columns[0]], df[columns[1]], df[columns[2]]])

# Heatmap erstellen
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', cbar=True, ax=ax)

ax.set_xlabel(columns[0])
ax.set_ylabel('Cluster Label')
plt.title('Cluster Heatmap')
plt.show()
