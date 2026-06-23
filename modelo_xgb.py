import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

df = pd.read_csv("csv/water_potability_limpo.csv")

# limites aceitáveis de potabilidade
df['ph_ideal'] = df['ph'].apply(lambda x: 1 if 6.5 <= x <= 8.5 else 0)
df['chloramines_seguro'] = df['Chloramines'].apply(lambda x: 1 if x <= 4.0 else 0)
df['sulfate_ideal'] = df['Sulfate'].apply(lambda x: 1 if x <= 250.0 else 0)
df['solids_seguro'] = df['Solids'].apply(lambda x: 1 if x <= 1000.0 else 0) # Limite comum de TDS

X = df.drop("Potability", axis=1)
y = df["Potability"]

# 3. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

peso_classe = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


xgb_base = XGBClassifier(
    random_state=42, 
    eval_metric='logloss',
    scale_pos_weight=peso_classe # peso equilibrado exato
)

# focada na curva ROC-AUC 
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}

# testar as combinações 
xgb_grid = GridSearchCV(
    estimator=xgb_base,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc', 
    n_jobs=-1,
    verbose=1
)

print(" (XGB) ")
xgb_grid.fit(X_train, y_train)

melhor_xgb = xgb_grid.best_estimator_
y_pred = melhor_xgb.predict(X_test)
y_prob = melhor_xgb.predict_proba(X_test)[:, 1] 

print("\n     Relatório de Classificação Definitivo    ")
print(classification_report(y_test, y_pred))

print("\n     Matriz de Confusão Definitiva     ")
print(confusion_matrix(y_test, y_pred))

print(f"\nROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")