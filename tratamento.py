import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("csv/water_potability.csv")


X = df.drop("Potability", axis=1)
y = df["Potability"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

knn_imputer = KNNImputer(n_neighbors=5)
X_imputed_knn = knn_imputer.fit_transform(X_scaled)


df_knn = pd.DataFrame(X_imputed_knn, columns=X.columns)
df_knn["Potability"] = y.values

print("Valores nulos após KNNImputer:\n", df_knn.isnull().sum())

df_knn.to_csv("csv/water_potability_limpo.csv", index=False)
print("Novo CSV salvo com sucesso!")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

X = df_knn.drop("Potability", axis=1)
y = df_knn["Potability"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definimos uma "grade" de configurações para o modelo testar
param_grid = {
    'n_estimators': [100, 200, 300, 500],       # Número de árvores
    'max_depth': [10, 15, 20, None],            # Profundidade máxima (evita overfitting)
    'min_samples_split': [2, 5, 10],            # Mínimo de amostras para dividir um nó
    'min_samples_leaf': [1, 2, 4],              # Mínimo de amostras em uma folha final
    'class_weight': ['balanced', 'balanced_subsample'] # Penaliza erros na classe minoritária
}

# criamos o modelo base
rf_base = RandomForestClassifier(random_state=42)

from sklearn.model_selection import RandomizedSearchCV

# Configuramos o buscador automático (testará 20 combinações diferentes aleatoriamente)
rf_random = RandomizedSearchCV(
    estimator=rf_base, 
    param_distributions=param_grid,
    n_iter=20, 
    cv=5, 
    verbose=1, 
    random_state=42, 
    n_jobs=-1, 
    scoring='f1'
)

print("Buscando a melhor configuração do Random Forest (isso pode levar alguns segundos)...")
rf_random.fit(X_train, y_train)

# Pegamos o melhor modelo encontrado
melhor_rf = rf_random.best_estimator_

print("\n✨ Melhores parâmetros encontrados:")
print(rf_random.best_params_)

# Avaliamos novamente com o modelo turbinado
y_pred_otimizado = melhor_rf.predict(X_test)

print("\n--- Relatório de Classificação (Otimizado) ---")
print(classification_report(y_test, y_pred_otimizado))

print("\n--- Matriz de Confusão (Otimizada) ---")
print(confusion_matrix(y_test, y_pred_otimizado))