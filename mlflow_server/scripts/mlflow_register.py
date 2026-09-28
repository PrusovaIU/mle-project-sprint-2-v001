import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
from mlflow.models import infer_signature

# 1. НАСТРОЙКИ ПОДКЛЮЧЕНИЯ
# Укажите адрес вашего MLflow-сервера
mlflow.set_tracking_uri("http://127.0.0.1:5000") 
mlflow.set_experiment("Building_Price_Prediction")

# 2. ЗАГРУЗКА ДАННЫХ И ПАРАМЕТРОВ
# Читаем params.yaml для получения гиперпараметров
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Читаем данные, которые DVC сгенерировал
# Важно: используем те же пути, что указаны в dvc.yaml
train_df = pd.read_csv("data/train.csv")
val_df = pd.read_csv("data/val.csv") # Используем для метрик (если нужно) или для примера

# Читаем модель, обученную DVC
# Для загрузки используем pickle, так как модель сохранена как .pkl
import pickle
with open("models/fitted_model.pkl", "rb") as f:
    model = pickle.load(f)

# Читаем метрики, вычисленные DVC
with open("cv_results/metrics.json", "r") as f:
    metrics = json.load(f)

# 3. ПОДГОТОВКА СИГНАТУРЫ
# Разделяем признаки и таргет
# В dvc.yaml target_col: price
target_col = params['data']['target_col'] # "price"
X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]

# Генерируем прогнозы на тренировочных данных для сигнатуры
predictions = model.predict(X_train)

# Инферим сигнатуру: какие колонки на входе, какой тип на выходе
signature = infer_signature(X_train, predictions)

# Создаем пример входа (первые 5 строк)
input_example = X_train.iloc[:5]

# 4. ЗАПУСК РАНА И ЛОГИРОВАНИЕ
with mlflow.start_run(run_name="DVC_Best_Model") as run:
    
    # -- Логирование Параметров --
    # Логируем параметры из params.yaml, которые относятся к модели
    mlflow.log_param("model_depth", params['model']['depth'])
    mlflow.log_param("model_iterations", params['model']['iterations'])
    mlflow.log_param("model_learning_rate", params['model']['learning_rate'])
    mlflow.log_param("model_loss_function", params['model']['loss_function'])
    
    # -- Логирование Метрик --
    # Логируем метрики из cv_results/metrics.json
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    # -- Логирование Артефактов (Файлов) --
    # Требование: "взять обученную модель, обучающие данные и метрики тестовой выборки"
    # Логируем сам файл модели
    mlflow.log_artifact("models/fitted_model.pkl", artifact_path="model_file")
    # Логируем файл обучающих данных (или их сэмпл, если он огромный)
    mlflow.log_artifact("data/train.csv", artifact_path="data")
    # Логируем файл метрик
    mlflow.log_artifact("cv_results/metrics.json", artifact_path="metrics")
    
    # -- Логирование и Регистрация Модели --
    # Регистрируем модель в Model Registry с сигнатурой
    mlflow.sklearn.log_model(
        sk_model=model,
        name="building_price_model", # artifact_path
        signature=signature,
        input_example=input_example,
        registered_model_name="BuildingPricePredictor" # Имя в Model Registry
    )

    print(f"Run ID: {run.info.run_id}")
    print("Модель успешно зарегистрирована в MLflow Model Registry.")
