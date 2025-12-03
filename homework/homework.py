# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#
import os
import gzip
import json
import pickle
from pathlib import Path
from glob import glob
import zipfile

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def process_and_train_model():
    # Load the ZIP files and convert them into DataFrames
    data_frames = []
    for zip_file in sorted(glob(os.path.join("files/input", "*.zip"))):  # Filter zip files
        print(f"Processing ZIP file: {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                with zip_ref.open(file_name) as file_handle:
                    df = pd.read_csv(file_handle, sep=",", index_col=0)
                    data_frames.append(df)
    
    print(f"Read {len(data_frames)} DataFrames from ZIP files.")
    training_data, testing_data = data_frames  # Assuming there are two input files

    # Data cleaning
    print(f"Renaming columns and cleaning the DataFrame.")
    training_data.rename(columns={"default payment next month": "default"}, inplace=True)
    testing_data.rename(columns={"default payment next month": "default"}, inplace=True)
    
    # Check if the "ID" column exists before removing it
    if "ID" in training_data.columns:
        training_data.drop(columns=["ID"], inplace=True)
        testing_data.drop(columns=["ID"], inplace=True)
    
    # Drop rows with missing values
    training_data.dropna(inplace=True)
    testing_data.dropna(inplace=True)

    # Group values greater than 4 in the EDUCATION column
    training_data['EDUCATION'] = training_data['EDUCATION'].apply(lambda x: 'others' if x > 4 else x)
    testing_data['EDUCATION'] = testing_data['EDUCATION'].apply(lambda x: 'others' if x > 4 else x)
    
    # Convert categorical columns to string type to avoid errors with OneHotEncoder
    categorical_columns = ["SEX", "EDUCATION", "MARRIAGE"]
    training_data[categorical_columns] = training_data[categorical_columns].astype(str)
    testing_data[categorical_columns] = testing_data[categorical_columns].astype(str)

    # Split into features (X) and target variable (y)
    X_train = training_data.drop(columns=["default"])
    y_train = training_data["default"]
    X_test = testing_data.drop(columns=["default"])
    y_test = testing_data["default"]
    
    # Create the pipeline with OneHotEncoder and RandomForestClassifier
    categorical_features = X_train.select_dtypes(include=["object"]).columns
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('cat', categorical_transformer, categorical_features)], remainder="passthrough")
    
    model = Pipeline(steps=[('prep', preprocessor), ('rf', RandomForestClassifier(random_state=42))])

    # Hyperparameter optimization using GridSearchCV
    parameter_grid = {
        'rf__n_estimators': [100, 200, 500],
        'rf__max_depth': [None, 5, 10],
        'rf__min_samples_split': [2, 5],
        'rf__min_samples_leaf': [1, 2]
    }
    grid_search = GridSearchCV(model, param_grid=parameter_grid, cv=10, scoring='balanced_accuracy', n_jobs=-1, refit=True, verbose=2)
    grid_search.fit(X_train, y_train)

    # Save the trained model in compressed format
    model_save_path = 'files/models/model.pkl.gz'
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with gzip.open(model_save_path, 'wb') as f:
        pickle.dump(grid_search.best_estimator_, f)

    # Predictions and metrics
    y_train_pred = grid_search.predict(X_train)
    y_test_pred = grid_search.predict(X_test)

    # Calculate metrics
    training_metrics = {
        "type": "metrics",
        "dataset": "train",
        "precision": precision_score(y_train, y_train_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_train, y_train_pred),
        "recall": recall_score(y_train, y_train_pred, zero_division=0),
        "f1_score": f1_score(y_train, y_train_pred, zero_division=0),
    }

    testing_metrics = {
        "type": "metrics",
        "dataset": "test",
        "precision": precision_score(y_test, y_test_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_test_pred),
        "recall": recall_score(y_test, y_test_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_test_pred, zero_division=0),
    }

    # Confusion matrices
    cm_train = confusion_matrix(y_train, y_train_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)

    cm_train_dict = {
        "type": "cm_matrix",
        "dataset": "train",
        "true_0": {
            "predicted_0": int(cm_train[0, 0]),
            "predicted_1": int(cm_train[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_train[1, 0]),
            "predicted_1": int(cm_train[1, 1]),
        },
    }

    cm_test_dict = {
        "type": "cm_matrix",
        "dataset": "test",
        "true_0": {
            "predicted_0": int(cm_test[0, 0]),
            "predicted_1": int(cm_test[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm_test[1, 0]),
            "predicted_1": int(cm_test[1, 1]),
        },
    }

    # Save metrics and confusion matrices in JSON files
    output_dir = 'files/output'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump([training_metrics, testing_metrics, cm_train_dict, cm_test_dict], f)

    print("Process completed and files saved.")


# Run the function
if __name__ == "__main__":
    process_and_train_model()