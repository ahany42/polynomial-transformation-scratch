import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations_with_replacement
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Lab Code PreProcessing Functions
def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X

def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X

df = pd.read_csv('dataset.csv')

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
df = Feature_Encoder(df, categorical_cols)
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
X_train_scaled = featureScaling(X_train, 0, 1)
X_test_scaled = featureScaling(X_test, 0, 1)

def poly_transform(X, degree):
    if not isinstance(X, np.ndarray):
        X = np.array(X)

    n_samples, n_features = X.shape
    result = [np.ones((n_samples, 1))]

    def generate_terms(d):
        if d > degree:
            return
        if d > 0:
            for comb in combinations_with_replacement(range(n_features), d):
                term = np.ones((X.shape[0], 1))
                for i in comb:
                    term *= X[:, i:i+1]
                result.append(term)
        generate_terms(d + 1)

    generate_terms(1)
    return np.hstack(result)

def poly_plot():
    degrees = range(1, 30)
    mse_train_values = []
    mse_test_values = []

    for d in degrees:
        X_poly_train = poly_transform(X_train_scaled, d)
        X_poly_test = poly_transform(X_test_scaled, d)
        # Lab Code Train And Predict Code
        model = LinearRegression()
        model.fit(X_poly_train, y_train)
        y_train_pred = model.predict(X_poly_train)
        y_test_pred = model.predict(X_poly_test)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_test_pred)
        mse_train_values.append(mse_train)
        mse_test_values.append(mse_test)
    plt.figure(figsize=(8, 5))
    plt.plot(degrees, mse_train_values, marker='o', linestyle='-', label="Training Error")
    plt.plot(degrees, mse_test_values, marker='s', linestyle='--', label="Test Error", color="red")
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Mean Squared Error')
    plt.title('MSE vs. Polynomial Degree')
    plt.legend()
    plt.grid(True)
    plt.show()
    
X_train_poly = poly_transform(X_train_scaled, degree=2)
print("Transformed Sample Data:\n", X_train_poly)
poly_plot()