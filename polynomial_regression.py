import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv('dataset.csv')
Y = df['RevenuePerDay']  
corr_matrix = df.corr()
top_features = corr_matrix.index[abs(corr_matrix['RevenuePerDay']) > 0.5].tolist()
top_features.remove('RevenuePerDay')  
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
X = df[top_features]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

def poly_transform(X, degree):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    n_samples, n_features = X.shape
    result = [np.ones((n_samples, 1))] 

    def generate_terms(d):
        if d > degree:
            return
        for comb in combinations_with_replacement(range(n_features), d):
            term = np.ones((X.shape[0], 1))  
            for i in comb:
                term *= X[:, i:i+1]  
            result.append(term)
        generate_terms(d + 1)  

    generate_terms(1) 
    return np.hstack(result)

def poly_plot():
    degrees = range(1,7)
    mse_values = []

    for d in degrees:
        X_poly_train = poly_transform(X_train, d)  
        X_poly_test = poly_transform(X_test, d)   
        # Lab Code to Train and Predict
        model = LinearRegression()  
        model.fit(X_poly_train, y_train)  
        y_pred = model.predict(X_poly_test)  

        mse = mean_squared_error(y_test, y_pred) 
        mse_values.append(mse)  
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, mse_values, marker='o', linestyle='-')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs. Polynomial Degree')
    plt.grid(True)
    plt.show()


X_train_poly = poly_transform(X_train, degree=2)
print("Transformed Sample Data:\n", X_train_poly)
poly_plot()
