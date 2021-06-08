from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
import pandas as pd

def importar_dataset():
    universalBank = pd.read_csv('dataset/UniversalBank.csv')
    X = universalBank.drop(['Personal Loan'], axis=1)
    y = universalBank['Personal Loan']

    return X, y

def train_logist():
    X, y = importar_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    X_trainE = scaler.fit_transform(X_train)
    X_testE = scaler.fit_transform(X_test)

    logistR = LogisticRegression()
    logistR.fit(X_trainE, y_train)

    y_pred = logistR.predict(X_testE)
    y_pred_train = logistR.predict(X_trainE)

    precision_test = accuracy_score(y_test, y_pred)
    precision_train = accuracy_score(y_train, y_pred_train)

    print(" =========================================================")
    print(" Regresion Logistica")
    print(" =========================================================")
    print(" Precision en la data de entrenamiento:", precision_train)
    print(" Precision en la data de prueba:", precision_test)
    print(" =========================================================")

if __name__ == "__main__":
    train_logist()