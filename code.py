# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# For inline plotting (if using Jupyter Notebook)
%matplotlib inline

# Load the dataset
df = pd.read_csv(r"C:\Users\jagru\Downloads\archive\TravelInsurancePrediction.csv")

# Drop unnecessary column (if present)
df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')  # Ignore if the column doesn't exist

# Replace missing values if any indicated as '-' or 'na'
df.replace(['-', 'na'], np.nan, inplace=True)

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Fill missing values with a strategy (e.g., median for numerical, mode for categorical)
# Fill missing values with a strategy (e.g., median for numerical, mode for categorical)
df['Age'] = df['Age'].fillna(df['Age'].median())
df['AnnualIncome'] = df['AnnualIncome'].fillna(df['AnnualIncome'].median())
df['FamilyMembers'] = df['FamilyMembers'].fillna(df['FamilyMembers'].mode()[0])
df['ChronicDiseases'] = df['ChronicDiseases'].fillna(df['ChronicDiseases'].mode()[0])


# Basic data info
print("\nData Info:")
print(df.info())

# Encode categorical variables
df["TravelInsurance"] = df["TravelInsurance"].map({0: "not purchased", 1: "purchased"})
df["GraduateOrNot"] = df["GraduateOrNot"].map({"No": 0, "Yes": 1})
df["FrequentFlyer"] = df["FrequentFlyer"].map({"No": 0, "Yes": 1})
df["EverTravelledAbroad"] = df["EverTravelledAbroad"].map({"No": 0, "Yes": 1})
df["Employment Type"] = df["Employment Type"].astype('category').cat.codes  # Label encoding

# Visualizations
px.histogram(df, x="Age", color="TravelInsurance", title="Travel Insurance vs Age").show()
px.histogram(df, x="Employment Type", color="TravelInsurance", title="Travel Insurance vs Employment Type").show()
px.histogram(df, x="AnnualIncome", color="TravelInsurance", title="Travel Insurance vs Annual Income").show()

# Check skewness of numeric columns
print("\nSkewness of numerical columns:\n", df.select_dtypes(include=['number']).skew())

# Correlation matrix (only numeric columns)
correlation = df.select_dtypes(include=['number']).corr()
print("\nCorrelation matrix:\n", correlation)

# Prepare features (X) and target (y)
df["TravelInsurance"] = df["TravelInsurance"].map({"not purchased": 0, "purchased": 1})  # Re-convert for model
X = df[["Age", "Employment Type", "GraduateOrNot", "AnnualIncome", "FamilyMembers",
        "ChronicDiseases", "FrequentFlyer", "EverTravelledAbroad"]]
y = df["TravelInsurance"]

# Split the data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.1, random_state=42)

# Train the model
model = DecisionTreeClassifier(random_state=42)
model.fit(xtrain, ytrain)

# Make predictions
predictions = model.predict(xtest)

# Accuracy and confusion matrix
print("\nModel Accuracy:", accuracy_score(ytest, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(ytest, predictions))

# Additional: Visualize the confusion matrix
sns.heatmap(confusion_matrix(ytest, predictions), annot=True, fmt='d', cmap='Blues', xticklabels=['Not Purchased', 'Purchased'], yticklabels=['Not Purchased', 'Purchased'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
