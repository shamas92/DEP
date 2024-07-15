import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the data
data = pd.read_csv('e:\\internships\\DEP\\Task 2\\Housing.csv')

# Data visualizations
# Plotting price distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True)
plt.title('Distribution of House Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Area vs Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['area'], y=data['price'])
plt.title('House Price vs Area')
plt.xlabel('Area (square feet)')
plt.ylabel('Price')
plt.show()

# Preparing data for the correlation matrix by selecting only numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Correlation matrix heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Separate the features and the target variable
X = data.drop('price', axis=1)
y = data['price']

# Define categorical columns for one-hot encoding
categorical_cols = X.select_dtypes(include=['object']).columns

# Create a ColumnTransformer for one-hot encoding of categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ],
    remainder='passthrough'
)

# Create a pipeline with preprocessing and the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model.fit(X_train, y_train)

# Predict the prices on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R^2 Score:", r2)
