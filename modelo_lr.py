import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("csv/water_potability_limpo.csv")
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# C: Inverso da força de regularização (menores = mais penalização)
# penalty: Tipo de penalidade (L1 tenta zerar variáveis inúteis, L2 encolhe todas)
# solver: Algoritmos de otimização adequados para L1 e L2
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'saga'] # Estes suportam tanto L1 quanto L2
}

# modelo base mantendo o balanceamento
lr_base = LogisticRegression(class_weight='balanced', random_state=42, max_iter=2000)

#  a busca
lr_grid = GridSearchCV(
    estimator=lr_base,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print(" (LR) ")
lr_grid.fit(X_train, y_train)

melhor_lr = lr_grid.best_estimator_
y_pred = melhor_lr.predict(X_test)

print("\n Melhores parâmetros encontrados para a LR:")
print(lr_grid.best_params_)

print("\n    Relatório de Classificação (LR Otimizada)    ")
print(classification_report(y_test, y_pred))

print("\n    Matriz de Confusão (LR Otimizada)    ")
print(confusion_matrix(y_test, y_pred))