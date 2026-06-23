import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("csv/water_potability_limpo.csv")
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# C: Controla a penalidade por erros (valores menores dão uma margem mais suave)
# gamma: Define o quão longe a influência de um único exemplo chega
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
    'kernel': ['rbf'] # O kernel RBF é ideal por conta da distribuição gaussiana dos gráficos
}
# balanceamento automático
svm_base = SVC(class_weight='balanced', random_state=42)

# focada no melhor F1-Score 
svm_grid = GridSearchCV(
    estimator=svm_base,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print(" (SVM) ")
svm_grid.fit(X_train, y_train)

# 5. Avaliar os resultados do campeão
melhor_svm = svm_grid.best_estimator_
y_pred = melhor_svm.predict(X_test)

print("\n Melhores parâmetros encontrados para o SVM:")
print(svm_grid.best_params_)

print("\n    Relatório de Classificação (SVM Otimizado) ")
print(classification_report(y_test, y_pred))

print("\n    Matriz de Confusão (SVM Otimizado)    ")
print(confusion_matrix(y_test, y_pred))