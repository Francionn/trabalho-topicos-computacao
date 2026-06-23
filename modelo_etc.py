import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv("csv/water_potability_limpo.csv")
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# max_depth: controla o crescimento para evitar que ele decore o ruído
# min_samples_split: força os nós a serem maiores antes de dividirem
# class_weight: testa o balanceamento padrão vs balanceamento por subamostra
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}

etc_base = ExtraTreesClassifier(random_state=42)

# focada no melhor F1-Score da Classe 1
etc_grid = GridSearchCV(
    estimator=etc_base,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

print(" (ETC)")
etc_grid.fit(X_train, y_train)

melhor_etc = etc_grid.best_estimator_
y_pred = melhor_etc.predict(X_test)

print("\nMelhores parâmetros encontrados para o ETC:")
print(etc_grid.best_params_)

print("\n    Relatório de Classificação (ETC Otimizado)    ")
print(classification_report(y_test, y_pred))

print("\n    Matriz de Confusão (ETC Otimizado)    ")
print(confusion_matrix(y_test, y_pred))