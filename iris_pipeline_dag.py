from datetime import datetime, timedelta
import requests
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

default_args = {
    'owner': 'shaifee',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def check_api_health():
    response = requests.get('http://127.0.0.1:8000/health')
    if response.status_code == 200:
        print("Health Check Passed:", response.json())
    else:
        raise Exception(f"Health check failed with status code {response.status_code}")

def make_iris_prediction():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    response = requests.post('http://127.0.0.1:8000/predict', json=payload)
    if response.status_code == 200:
        print("Prediction Response:", response.json())
    else:
        raise Exception(f"Prediction request failed: {response.text}")

with DAG(
    'iris_model_pipeline',
    default_args=default_args,
    description='Automated health check and prediction pipeline for Iris API',
    schedule='@daily',
    catchup=False,
) as dag:

    health_task = PythonOperator(
        task_id='check_api_health',
        python_callable=check_api_health,
    )

    predict_task = PythonOperator(
        task_id='make_iris_prediction',
        python_callable=make_iris_prediction,
    )

    health_task >> predict_task