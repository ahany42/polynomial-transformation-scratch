import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math

df = pd.read_csv('dataset.csv')
Y = df['RevenuePerDay']  
corr_matrix = df.corr()
top_features = corr_matrix.index[abs(corr_matrix['RevenuePerDay']) > 0.5].tolist()
top_features.remove('RevenuePerDay')  
plt.figure(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
X = df[top_features]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)


def polynomial_transform(degree,features):
#poly rule
#sum n+k-1 ncr k where starts from 0 to d (degree)
    n = len(features)
    d = degree
    total_combinations = 0
    for k in range(d+1):
        total_combinations +=  math.comb((n+k-1), k) 
    print(total_combinations)
    
polynomial_transform(10,top_features)