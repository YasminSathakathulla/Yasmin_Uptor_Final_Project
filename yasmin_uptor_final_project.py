import pandas as pd
#from sklearn.datasets import load_boston #sample dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load Boston Housing dataset
boston=pd.read_csv("boston_house_prices.csv")
#X = data.drop("target_variable", axis=1)  # Features
#y = data["target_variable"]  # Target variable

print(boston.columns)

#X=
#y=

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Standardize the data
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)
#
# # Apply PCA
# pca = PCA(n_components=0.95)  # Keep components explaining 95% of variance
# X_train_pca = pca.fit_transform(X_train_scaled)
# X_test_pca = pca.transform(X_test_scaled)
#
# # Create and fit linear regression model
# model = LinearRegression()
# model.fit(X_train_pca, y_train)
#
# # Make predictions
# y_pred = model.predict(X_test_pca)
#
# # Evaluate model performance
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)
