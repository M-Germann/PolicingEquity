import pandas as pd
import matplotlib.pyplot as plt

# Loading process

# Path to both chosen files
file1 = r'E:\Dropbox\Studium\Unsupervised Maschine Learning\Data Task 2\archive\Chosen\23-00089_UOF-P.csv'
file2 = r'E:\Dropbox\Studium\Unsupervised Maschine Learning\Data Task 2\archive\Chosen\37-00027_UOF-P_2014-2016_prepped.csv'

# Load files as data frames:
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Combine data frames
df = pd.concat([df1, df2])

# Diagram
fig, ax = plt.subplots()
df[['OFFICER_RACE', 'SUBJECT_RACE']].value_counts().plot(kind='bar', ax=ax)

# Shorten labels
ax.set_xticklabels([f"{x[0][0]}-{x[1][0]}" for x in df[['OFFICER_RACE', 'SUBJECT_RACE']].value_counts().index])

# Labels and title
plt.xlabel('Combination')
plt.ylabel('Count')
plt.title('Count OFFICER_RACE to SUBJECT_RACE')

# Show
plt.show()
