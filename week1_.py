import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 1. Load a dataset using Pandas and summarize basic stats
df_titanic = pd.read_csv('titanic.csv')

print("--- Titanic Dataset Info ---")
df_titanic.info()
print("\n--- Titanic Dataset Description ---")
print(df_titanic.describe())

# 2. Handle missing data using mean/median imputation.
# Impute 'Age' with the median and 'Embarked' with the most frequent value (mode)
df_titanic['Age'].fillna(df_titanic['Age'].median(), inplace=True)
df_titanic['Embarked'].fillna(df_titanic['Embarked'].mode()[0], inplace=True)

# 3. Encode categorical variables using LabelEncoder & OneHotEncoder.
# Apply LabelEncoder to the 'Sex' column
le = LabelEncoder()
df_titanic['Sex'] = le.fit_transform(df_titanic['Sex'])

# Apply OneHotEncoder to the 'Embarked' column
ohe = OneHotEncoder(sparse_output=False)
embarked_encoded = ohe.fit_transform(df_titanic[['Embarked']])

# Convert the one-hot encoded array back into a DataFrame and concatenate it
ohe_columns = ohe.get_feature_names_out(['Embarked'])
embarked_df = pd.DataFrame(embarked_encoded, columns=ohe_columns)
df_titanic = pd.concat([df_titanic.drop('Embarked', axis=1), embarked_df], axis=1)

# Mini Project Task: Visualize age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df_titanic['Age'], bins=30, kde=True, color='skyblue')
plt.title('Titanic: Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Output cleaned dataset as new CSV
cleaned_filename = 'cleaned_titanic.csv'
df_titanic.to_csv(cleaned_filename, index=False)
print(f"\nCleaned dataset successfully saved to {cleaned_filename}")