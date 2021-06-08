from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import pandas as pd

def importar_dataset():
    universalBank = pd.read_csv('dataset/UniversalBank.csv')
    X = universalBank.drop(['Personal Loan'], axis=1)
    y = universalBank['Personal Loan']

    return X, y

def decision_tree():
    
    X, y = importar_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    decisionTC = DecisionTreeClassifier()
    decisionTC.fit(X_train, y_train)

    y_pred = decisionTC.predict(X_test)
    y_pred_train = decisionTC.predict(X_train)

    precision_test = accuracy_score(y_test, y_pred)
    precision_train = accuracy_score(y_train, y_pred_train)
    
    print("=========================================================")
    print("Decision Tree Classification")
    print("=========================================================")
    print("Precision en la data de entrenamiento:", precision_train)
    print("Precision en la data de prueba:", precision_test)
    print("=========================================================")

if __name__ == "__main__":
    decision_tree()