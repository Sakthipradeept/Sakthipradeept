import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('your_dataset.csv')

# Data cleaning (if necessary)
data = data.dropna()  # Drop rows with missing values

# Split features and target variable
X = data.drop(columns=['target_column'])
y = data['target_column']

# Feature scaling (if necessary)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Now X_train, X_test, y_train, y_test are ready for training your machine learning model
