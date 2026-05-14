import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

df = pd.read_csv('adult_data_train.csv')

#delate columns education - marital
df_cleaned = df.drop(columns=['education', 'marital-status'])
print(df_cleaned.head())

#we get the types of all the collumns/ получаем типы всех столбцов
data_types = df_cleaned.dtypes
#we count how many are type object and numeric / подсчитаем, склоько из них имеют тип object
non_numeric_cols = df_cleaned.select_dtypes(include=['object']).columns
non_numeric_count = len(non_numeric_cols)

numeric_count = (data_types == 'int64').sum() + (data_types == 'float64').sum()

print(f"object type:{non_numeric_count}")
print(f"numeric type:{numeric_count}")

ax = df['label'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'] ,edgecolor='black')
#construct the histograma of the class distribution
#label is the column that contains the classes (0 and 1)
ax.bar_label(ax.containers[0], label_type='edge', padding=3)

plt.title('Distribution of Objects by class')
plt.xlabel('class (0 or 1)')
plt.ylabel('Number of Objects')
plt.xticks(rotation=0)
plt.ylim(0, df['label'].value_counts().max()*1.1)
plt.show()
#3. Calculate the proportion of objects in class 0
proportions = df['label'].value_counts(normalize=True)


class_0_proportion =round(proportions[0], 3)
print(f"Proportion of objects in class 0:{class_0_proportion}")

numeric_cols=df.select_dtypes(include=['number']).columns
df_numeric = df[numeric_cols].copy()

#Convert labels 1 and 0 to treu and False as requested
df_numeric['label'] = df_numeric['label'].astype(bool)

#Separate the features (X) from the target label
X = df_numeric.drop(columns=['label'])
y = df_numeric['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=53, stratify=y
)

#Find sample mean for the column fnlwgt of the obtained training dataset
fnlwgt_mean = X_train['fnlwgt'].mean()
print(f"Sample mean for fnlwgt (Train): {round(fnlwgt_mean,3)}")

knn = KNeighborsClassifier()
knn.fit(X_train, y_train) #Training the model

#Evaluate on the test data
y_pred = knn.predict(X_test)

f1 = f1_score(y_test, y_pred)
print(f"F1 Score for test dataset: {round(f1,3)}")