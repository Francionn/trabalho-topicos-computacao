import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import mlflow
import mlflow.sklearn

df = pd.read_csv("csv/water_potability_limpo.csv")
X = df.drop("Potability", axis=1)
y = df["Potability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# n_estimators: Número de árvores na floresta
# max_depth: Limita a profundidade para evitar o sobreajuste (overfitting) no ruído
# min_samples_split: Número mínimo de amostras necessárias para dividir um nó interno
# class_weight: Estratégias de pesos para lidar com o desbalanceamento das classes
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'class_weight': ['balanced', 'balanced_subsample']
}

rf_base = RandomForestClassifier(random_state=42)

#  buscador automático f
rf_grid = GridSearchCV(
    estimator=rf_base,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

mlflow.set_experiment("benchmarking_classificadores")

with mlflow.start_run(run_name="Random Forest"):
    print(" (RF) ")
    rf_grid.fit(X_train, y_train)

    melhor_rf = rf_grid.best_estimator_
    y_pred = melhor_rf.predict(X_test)

    print("\n Melhores parâmetros encontrados para o RF:")
    print(rf_grid.best_params_)
    mlflow.log_params(rf_grid.best_params_)

    model_info = mlflow.sklearn.log_model(
        sk_model=melhor_rf, 
        name="modelo RF"
    )

    eval_data = X_test.copy()
    eval_data["Potability"] = y_test

    evaluation_result = mlflow.evaluate(
        model=model_info.model_uri,
        data=eval_data,
        targets="Potability",
        model_type="classifier",
    )

    print("\n    Relatório de Classificação (RF Otimizado) ")
    print(classification_report(y_test, y_pred))

    print("\n    Matriz de Confusão (RF Otimizado)    ")
    print(confusion_matrix(y_test, y_pred))