import numpy as np
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Generowanie prostego zbioru danych
def generate_data():

    cloudAlpha = [2 * [0] for _ in range(100)]
    cloudBeta = [2 * [0] for _ in range(100)]
    for i in range(100):
        for j in range(2):
            cloudAlpha[i][j] = randint(0, 10)
            cloudBeta[i][j] = randint(50, 60)

    data = np.vstack((cloudAlpha, cloudBeta))
    labels = np.hstack((np.zeros(100), np.ones(100)))

    return data, labels



# Trenowanie prostego modelu regresji logistycznej
def train_model():
    # Generowanie danych
    data, labels = generate_data()

    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Trenowanie modelu
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predykcja na zbiorze testowym
    y_pred = model.predict(X_test)

    # Wyliczenie dokładności
    accuracy = accuracy_score(y_test, y_pred)

    # Zapis wyniku
    with open('accuracy.txt', 'w') as f:
        f.write(f"Model trained with accuracy: {accuracy * 100:.2f}%\n")

    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    train_model()
