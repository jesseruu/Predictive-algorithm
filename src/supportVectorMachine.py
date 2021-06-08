from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 
import numpy as np
import pandas as pd

def importar_dataset():
    universalBank = pd.read_csv('dataset/UniversalBank.csv')
    X = universalBank.drop(['Personal Loan'], axis=1)
    y = universalBank['Personal Loan']

    return X, y

def support_vectoM():

    X, y = importar_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    supporV = svm.SVC()
    supporV.fit(X_train, y_train)

    y_pred = supporV.predict(X_test)
    y_pred_train = supporV.predict(X_train)

    precision_test = accuracy_score(y_test, y_pred)
    precision_train = accuracy_score(y_train, y_pred_train)
    
    print("=========================================================")
    print("Support Vector Machine")
    print("=========================================================")
    print("Precision en la data de entrenamiento:", precision_train)
    print("Precision en la data de prueba:", precision_test)
    print("=========================================================")

if __name__ == "__main__":
    support_vectoM()