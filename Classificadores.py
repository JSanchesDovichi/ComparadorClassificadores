import time
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def executar_teste(classificador, X_treino, y_treino, X_teste, y_teste):
    print("Classificador: ", classificador.__name__) 

    start = time.process_time()
    clf = classificador.etapa_fit(X_treino, y_treino)
    end = time.process_time()

    tempo_fitting = (end - start) * 10**3

    print("Tempo decorrido (fitting): ", tempo_fitting, "ms.")

    start = time.process_time()
    y_pred = classificador.etapa_predicao(clf, X_teste)
    end = time.process_time()

    tempo_predicao = (end - start) * 10**3

    print("Tempo decorrido (predicao): ", tempo_predicao, "ms.")

    acc = clf.score(X_teste, y_teste) * 100

    print("Acurácia: ", f'{acc:.2f}', "%")

    cm = confusion_matrix(y_teste, y_pred, labels=clf.classes_)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Não\ndiabético", "Diabético"])

    disp.plot()

    plt.savefig('matriz_confusao' + classificador.__name__ + '.png')

    print("---")

    return tempo_fitting, tempo_predicao, acc

class KNN:
    def executar_teste():
        print("TESTE ")

    def etapa_fit(X_treino, y_treino):
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_neighbors=3).fit(X_treino, y_treino)

    def etapa_predicao(clf, X_teste):
        return clf.predict(X_teste)



class AdaBoost:
    def executar_teste():
        print("TESTE ")

    def etapa_fit(X_treino, y_treino):
        from sklearn.ensemble import AdaBoostClassifier
        
        return AdaBoostClassifier(n_estimators=100, random_state=0).fit(X_treino, y_treino)

    def etapa_predicao(clf, X_teste):
        return clf.predict(X_teste)

class RandomForest:
    def executar_teste():
        print("TESTE ")

    def etapa_fit(X_treino, y_treino):
        from sklearn.ensemble import RandomForestClassifier
        
        return RandomForestClassifier(max_depth=2, random_state=0).fit(X_treino, y_treino)

    def etapa_predicao(clf, X_teste):
        return clf.predict(X_teste)

class MLP:
    def executar_teste():
        print("TESTE ")

    def etapa_fit(X_treino, y_treino):
        from sklearn.neural_network import MLPClassifier
        
        return MLPClassifier(random_state=1, max_iter=300).fit(X_treino, y_treino)

    def etapa_predicao(clf, X_teste):
        return clf.predict(X_teste)
   