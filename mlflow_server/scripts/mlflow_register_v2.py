import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import yaml
from dotenv import load_dotenv, find_dotenv
from os import getenv, path


# Параметры .env:
load_dotenv(find_dotenv())
EXPERIMENT_NAME = getenv('EXPERIMENT_NAME', 'buildings_flats_price_prediction')
MODEL_NAME = getenv('MODEL_NAME', 'buildings_flats_price_model')
MLFLOW_URI = getenv('MLFLOW_URI', 'http://localhost:5000')

# Концигурация
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Загрузка артефактов, сохраненных из ноутбука:
ARTIFACT_DIR = 'model_improvement/artifacts_v2'

## Пайплайн:
pipeline = joblib.load(path.join(ARTIFACT_DIR, 'pipeline.joblib'))

## Сигнатура:
X_val = pd.read_csv(path.join(ARTIFACT_DIR, 'X_val.csv'))
signature = infer_signature(X_val, pipeline.predict(X_val))
input_example = X_val.head(1)

## Метрики:
with open(path.join(ARTIFACT_DIR, 'metrics.json'), 'r') as f:
    metrics = json.load(f)

## Параметры модели:
model_params = pipeline.named_steps['model'].get_params()


# Запуск MLflow и логирование
with mlflow.start_run(run_name='v2_autofeat_pipeline_model') as run:
    # Логируем параметры модели:
    mlflow.log_param('model_type', 'CatBoostRegressor')
    mlflow.log_param('pipeline_steps', list(pipeline.named_steps.keys()))
    mlflow.log_param('model_depth', model_params.get('depth'))
    mlflow.log_param('model_iterations', model_params.get('iterations'))
    mlflow.log_param('model_learning_rate', model_params.get('learning_rate'))
    mlflow.log_param('model_loss_function', model_params.get('loss_function'))
    mlflow.log_param('random_state', model_params.get('random_state'))
    mlflow.log_param('target_transformation', 'log1p')

    # Логируем метрики валидации (по одной)
    for key, value in metrics.items():
        mlflow.log_metric(key, value)
    
    # Логируем ноутбук с EDA и выводами
    mlflow.log_artifact('model_improvement/notebook.ipynb', artifact_path='eda')
    
    # Логируем сохраненные данные и метрики
    mlflow.log_artifact(path.join(ARTIFACT_DIR, 'metrics.json'), artifact_path='validation')
    mlflow.log_artifact(path.join(ARTIFACT_DIR, 'X_val.csv'), artifact_path='validation')
    
    # Регистрируем весь пайплайн
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path='model',
        signature=signature,
        input_example=input_example,
        registered_model_name='buildings_flats_price_model', 
    )

    print(f'Run ID: {run.info.run_id}')
    print(f'Зарегистрирована модель: {MODEL_NAME}')
