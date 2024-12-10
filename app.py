import pandas as pd
from py_irt.models.one_param_logistic import OneParamLog  # Modelo 1PL
from py_irt.models.two_param_logistic import TwoParamLog  # Modelo 2PL
from py_irt.models.three_param_logistic import ThreeParamLog  # Modelo 3PL
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import pyro
import app1P
import sys
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer
from py_irt.dataset import Dataset

# Configuración de la página
st.set_page_config(page_title="Análisis", layout="wide")

# Título
st.title("Análisis")

# Cargar archivo
st.sidebar.header("Sube tus datos")
uploaded_file = st.sidebar.file_uploader("Selecciona un archivo CSV", type=["csv"])

if uploaded_file:
    # Leer los datos cargados
    data = pd.read_csv(uploaded_file)

    # Verificación de la estructura del archivo cargado
    if "Persona" not in data.columns or data.shape[1] < 2:
        st.error("El archivo debe contener una columna 'Persona' y al menos un ítem.")
    else:
        # Mostrar los datos cargados
        st.write("**Datos cargados:**")
        st.dataframe(data)

        # Preparamos la matriz de respuestas para el modelo (excluyendo la columna 'Persona')
        response_matrix = data.drop(columns=["Persona"]).values
        num_subjects, num_items = response_matrix.shape

        # Selección del modelo
        st.sidebar.header("Configuración del Modelo")
        model_type = st.sidebar.radio("Selecciona el modelo TRI:", ["1PL", "2PL", "3PL"])

       # Definir priors comunes
        if model_type == "1PL":
           app1P.main(response_matrix)     
           sys.exit()
        elif model_type == "2PL":
            priors = {
                "ability": {"dist": "normal", "mean": 0, "std": 1},
                "diff": {"dist": "normal", "mean": 0, "std": 1},
                "disc": {"dist": "normal", "mean": 0, "std": 1}
            }
            model = TwoParamLog(num_items=num_items, num_subjects=num_subjects, device="cpu", priors=priors)
        elif model_type == "3PL":
            priors = {
                "ability": {"dist": "normal", "mean": 0, "std": 1},
                "diff": {"dist": "normal", "mean": 0, "std": 1},
                "disc": {"dist": "lognormal", "mean": 0, "std": 1},
                "lambdas": {"dist": "beta", "alpha": 1, "beta": 1}
            }
            model = ThreeParamLog(num_items=num_items, num_subjects=num_subjects, device="cpu", priors=priors)

        # Set random seed for reproducibility
        pyro.set_rng_seed(0)
        pyro.clear_param_store()

        # Crear tensores con dimensiones correctas
        response_data = torch.tensor(response_matrix, device=model.device, dtype=torch.float)
        response_data = response_data.reshape(num_subjects * num_items)

        subjects = torch.repeat_interleave(torch.arange(num_subjects), num_items)
        items = torch.tile(torch.arange(num_items), (num_subjects,))

        # Inicializar entrenamiento
        optimizer = Adam({"lr": 0.01})
        svi = SVI(model.get_model(), 
                  model.get_guide(),
                  optimizer, 
                  loss=Trace_ELBO())

        # Entrenar el modelo
        num_iterations = 3000
        for i in range(num_iterations):
            loss = svi.step(subjects, items, response_data)
            if i % 100 == 0:
                print(f'Iteration {i} Loss: {loss}')
        
        # Obtener todos los parámetros usando export()
        params = model.export()

        # Mostrar parámetros dependiendo del modelo
        item_indices = list(range(len(params["diff"])))

        st.write(f"**Parámetros estimados del modelo {model_type}:**")
        if model_type == "1PL":
            params_df = pd.DataFrame({
                "Ítem": item_indices,
                "Dificultad (b)": params["diff"]
            })
        elif model_type == "2PL":
            params_df = pd.DataFrame({
                "Ítem": item_indices,
                "Discriminación (a)": params["disc"],
                "Dificultad (b)": params["diff"]
            })
        elif model_type == "3PL":
            params_df = pd.DataFrame({
                "Ítem": item_indices,
                "Discriminación (a)": params["disc"],
                "Dificultad (b)": params["diff"],
                "Adivinanza (c)": params["lambdas"]
            })
        st.dataframe(params_df)

        # Visualizar las curvas características de los ítems (ICC)
        st.write("**Curvas Características de los Ítems:**")
        fig, ax = plt.subplots(figsize=(10, 6))

        # Definir el rango de valores de theta (rasgo latente)
        theta = np.linspace(-3, 3, 100)

        # Generar las curvas características para cada ítem
        for i in range(len(item_indices)):
            if model_type == "1PL":
                p = 1 / (1 + np.exp(-(theta - params["diff"][i])))  # 1PL
            elif model_type == "2PL":
                p = 1 / (1 + np.exp(-(params["disc"][i] * (theta - params["diff"][i]))))  # 2PL
            elif model_type == "3PL":
                p = params["lambdas"][i] + (1 - params["lambdas"][i]) / (1 + np.exp(-(params["disc"][i] * (theta - params["diff"][i]))))  # 3PL
            ax.plot(theta, p, label=f"Ítem {i+1}")

        ax.set_xlabel("Theta")
        ax.set_ylabel("Probabilidad de respuesta correcta")
        ax.set_title("Curvas Características de los Ítems")
        ax.legend()
        st.pyplot(fig)

        # Permitir la descarga de los resultados
        st.download_button(
            label="Descargar resultados calculados",
            data=params_df.to_csv(index=False).encode('utf-8'),
            file_name=f"resultados_calculados_{model_type}.csv",
            mime="text/csv"
        )
