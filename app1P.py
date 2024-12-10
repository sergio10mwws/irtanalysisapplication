import pandas as pd
from py_irt.models.one_param_logistic import OneParamLog  # Modelo 1PL
from py_irt.models.two_param_logistic import TwoParamLog  # Modelo 2PL
from py_irt.models.three_param_logistic import ThreeParamLog  # Modelo 3PL
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer
from py_irt.dataset import Dataset

def main(response_matrix):
    num_subjects, num_items = response_matrix.shape
    subjects = []
    items = []
    observations = []
    training_example = []

    for i in range(response_matrix.shape[0]):
        for j in range(response_matrix.shape[1]):
            subjects.append(i)
            items.append(j)
            observations.append(float(response_matrix[i,j]))
            training_example.append(True)

            # Create mappings
            ix_to_subject_id = {i:str(i) for i in range(response_matrix.shape[0])}
            ix_to_item_id = {i:str(i) for i in range(response_matrix.shape[1])}
            subject_id_to_ix = {str(i):i for i in range(response_matrix.shape[0])}
            item_id_to_ix = {str(i):i for i in range(response_matrix.shape[1])}

    # Create dataset object with all required fields
    dataset = Dataset(
        observation_subjects=subjects,
        observation_items=items, 
        observations=observations,
        training_example=training_example,
        ix_to_subject_id=ix_to_subject_id,
        ix_to_item_id=ix_to_item_id,
        subject_ids=list(subject_id_to_ix.keys()),
        item_ids=list(item_id_to_ix.keys()),
        subject_id_to_ix=subject_id_to_ix,
        item_id_to_ix=item_id_to_ix
    )

    config = IrtConfig(model_type='1pl', log_every=500, dropout=.2)
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)
    trainer.train(epochs=100, device='cpu')        

    # Set random seed for reproducibility
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    # First initialize the trainer
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)

    # Train the model using CPU
    trainer.train(epochs=100, device='cpu')

    # Crear tensores con dimensiones correctas
    response_data = torch.tensor(response_matrix, device=trainer.irt_model.device, dtype=torch.float)
    response_data = response_data.reshape(num_subjects * num_items)

    subjects = torch.repeat_interleave(torch.arange(num_subjects), num_items)
    items = torch.tile(torch.arange(num_items), (num_subjects,))

     # Establecer semilla aleatoria para reproducibilidad
    pyro.set_rng_seed(0)
    pyro.clear_param_store()

    # Inicializar el entrenador nuevamente
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)
    trainer.train(epochs=5000, device='cpu')  # Entrenar en CPU

    # Crear tensores con dimensiones correctas
    response_data = torch.tensor(response_matrix, device=trainer.irt_model.device, dtype=torch.float)
    response_data = response_data.reshape(num_subjects * num_items)

    # Obtener parámetros utilizando export()
    params = trainer.irt_model.export()

    # Mostrar los parámetros estimados del modelo
    item_indices = list(range(len(params["diff"])))

    st.write(f"**Parámetros estimados del modelo {config.model_type}:**")

    params_df = pd.DataFrame({
        "Ítem": item_indices,
        "Dificultad (b)": params["diff"]
    })

    st.dataframe(params_df)

    # Calcular las puntuaciones (habilidades) de los sujetos
    abilities = params["ability"]  # Extraemos las puntuaciones de habilidad para los sujetos

    # Transformación de las puntuaciones theta a una escala de 0-10
    min_theta = np.min(abilities)  # Puntuación mínima de habilidad
    max_theta = np.max(abilities)  # Puntuación máxima de habilidad

    transformed_abilities = 10 * (abilities - min_theta) / (max_theta - min_theta)

    # Mostrar las puntuaciones transformadas (en la escala de 0-10)
    st.write("**Puntuaciones de Habilidad transformadas (escala 0-10):**")
    ability_df = pd.DataFrame({
        "Sujeto": list(range(num_subjects)),
        "Habilidad (Theta)": abilities,
        "Puntuación transformada (0-10)": transformed_abilities
    })

    st.dataframe(ability_df)

    # Descargar las puntuaciones de habilidad transformadas
    st.download_button(
        label="Descargar puntuaciones de habilidad transformadas (escala 0-10)",
        data=ability_df.to_csv(index=False).encode('utf-8'),
        file_name="puntuaciones_habilidad_transformadas.csv",
        mime="text/csv"
    )

    # Visualizar las curvas características de los ítems (ICC)
    st.write("**Curvas Características de los Ítems (ICC):**")
    fig, ax = plt.subplots(figsize=(10, 6))

    # Definir el rango de valores de theta (rasgo latente)
    theta = np.linspace(-3, 3, 100)

    # Generar las curvas características para cada ítem
    for i in range(num_items):
        p = 1 / (1 + np.exp(-(theta - params["diff"][i])))  # 1PL
        ax.plot(theta, p, label=f"Ítem {i}")

    ax.set_xlabel("Theta")
    ax.set_ylabel("Probabilidad de respuesta correcta")
    ax.set_title("Curvas Características de los Ítems (ICC)")
    ax.legend()
    st.pyplot(fig)

    # Permitir la descarga de los resultados
    st.download_button(
        label="Descargar resultados calculados",
        data=params_df.to_csv(index=False).encode('utf-8'),
        file_name=f"resultados_calculados_{config.model_type}.csv",
        mime="text/csv"
    )
            
            # Obtener todos los parámetros usando export()
            # params = model.export()

            # # Mostrar parámetros dependiendo del modelo
            # item_indices = list(range(len(params["diff"])))

            # st.write(f"**Parámetros estimados del modelo {model_type}:**")
            
            # params_df = pd.DataFrame({
            #     "Ítem": item_indices,
            #     "Dificultad (b)": params["diff"]
            # })
        
            # st.dataframe(params_df)

            # # Visualizar las curvas características de los ítems (ICC)
            # st.write("**Curvas Características de los Ítems:**")
            # fig, ax = plt.subplots(figsize=(10, 6))

            # # Definir el rango de valores de theta (rasgo latente)
            # theta = np.linspace(-3, 3, 100)

            # # Generar las curvas características para cada ítem

            # p = 1 / (1 + np.exp(-(theta - params["diff"][i])))  # 1PL

            # ax.set_xlabel("Theta")
            # ax.set_ylabel("Probabilidad de respuesta correcta")
            # ax.set_title("Curvas Características de los Ítems")
            # ax.legend()
            # st.pyplot(fig)

            # # Permitir la descarga de los resultados
            # st.download_button(
            #     label="Descargar resultados calculados",
            #     data=params_df.to_csv(index=False).encode('utf-8'),
            #     file_name=f"resultados_calculados_{model_type}.csv",
            #     mime="text/csv"
            # )
