import pandas

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt

def preparar_dataset(dataframe):
    # Etapa de preparação das labels do dataset.
    # Não é necessário se os dados forem numéricos.
    cols = ['gender', 'smoking_history']
    dataframe[cols] = dataframe[cols].apply(LabelEncoder().fit_transform)

    colunas = dataframe.columns.drop('diabetes')

    X = dataframe[colunas].values

    le = LabelEncoder()
    y = le.fit_transform(dataframe['diabetes'])

    return X,y

def ler_dataset_csv(nome_dataset_csv):
    # Lendo um dataset em formato CSV com um nome de 
    # arquivo fornecido.
    dataframe = pandas.read_csv(nome_dataset_csv)

    # Caso seja necessário preparar o dataset:
    return preparar_dataset(dataframe)

def separar_dataset(X, y):
    # Separando o dataset em elementos para treinamento
    # e testes. A proporção padrão é 70/30.
    # Elementos retornados: X_treino, X_teste, y_treino, y_teste.
    return train_test_split(X, y, train_size=0.7, test_size=0.3)


def detectar_outliers(X):
    clf = LocalOutlierFactor(n_neighbors=3)
    scores = clf.fit_predict(X)

    labels = ["Outliers", "Inliners"]
    resultados = []

    n_outliers = 0
    n_inliners = 0

    for number in scores:
        if number == 1:
            n_inliners+=1
        else:
            n_outliers+=1

    resultados.append(n_outliers)
    resultados.append(n_inliners)

    print("Outliners detectados: ", n_outliers)
    print("Inliners detectados: ", n_inliners)

    print("---")

    fig,ax = plt.subplots()

    for i in range(len(labels)):
        plt.text(i, resultados[i], resultados[i], ha='center', va='bottom')

    ax.bar(labels, resultados)

    plt.savefig('outliers.png')
