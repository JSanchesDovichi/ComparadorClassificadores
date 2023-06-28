import dataset_utils
import Classificadores
from Classificadores import KNN, AdaBoost, RandomForest, MLP

X, y = dataset_utils.ler_dataset_csv('diabetes_prediction_dataset.csv')

dataset_utils.detectar_outliers(X)

X_treino, X_teste, y_treino, y_teste = dataset_utils.separar_dataset(X, y)

classificadores = ["KNN", "AdaBoost", "Random Forest", "MLP",]
tempos_fitting = []
tempos_predicao = []
acuracias = []

fitting_knn, pred_knn, acc_knn = Classificadores.executar_teste(KNN, X_treino, y_treino, X_teste, y_teste)

tempos_fitting.append(fitting_knn)
tempos_predicao.append(pred_knn)
acuracias.append(acc_knn)

fitting_adaboost, pred_adaboost, acc_adaboost = Classificadores.executar_teste(AdaBoost, X_treino, y_treino, X_teste, y_teste)

tempos_fitting.append(fitting_adaboost)
tempos_predicao.append(pred_adaboost)
acuracias.append(acc_adaboost)

fitting_randomforest, pred_randomforest, acc_randomforest = Classificadores.executar_teste(RandomForest, X_treino, y_treino, X_teste, y_teste)

tempos_fitting.append(fitting_randomforest)
tempos_predicao.append(pred_randomforest)
acuracias.append(acc_randomforest)

fitting_mlp, pred_mlp, acc_mlp = Classificadores.executar_teste(MLP, X_treino, y_treino, X_teste, y_teste)

tempos_fitting.append(fitting_mlp)
tempos_predicao.append(pred_mlp)
acuracias.append(acc_mlp)

fig,ax = plt.subplots()

ax.bar(classificadores, tempos_fitting)

for i in range(len(classificadores)):
    plt.text(i, tempos_fitting[i], f'{tempos_fitting[i]:.2f}', ha='center',va='bottom')
    # plt.text(i, tempos_fitting[i], tempos_fitting[i], ha='center',va='bottom')

plt.savefig('tempos_fitting.png')

plt.close(fig)

fig,ax = plt.subplots()

ax.set_ylabel('Tempo (ms)')

ax.bar(classificadores, tempos_predicao)

for i in range(len(classificadores)):
    plt.text(i, tempos_predicao[i], f'{tempos_predicao[i]:.2f}', ha='center',va='bottom')
    # plt.text(i, tempos_predicao[i], tempos_predicao[i], ha='center',va='bottom')

plt.savefig('tempos_predicao.png')

plt.close(fig)

fig,ax = plt.subplots()

ax.set_ylabel('Tempo (ms)')

ax.bar(classificadores, acuracias)

for i in range(len(classificadores)):
    plt.text(i, acuracias[i], f'{acuracias[i]:.2f}', ha='center', va='bottom')
    # plt.text(i, acuracias[i], acuracias[i], ha='center', va='bottom')

plt.savefig('acuracias.png')