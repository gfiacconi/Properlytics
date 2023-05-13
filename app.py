import os 
import time
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st 
import plotly.express as px
import matplotlib.pyplot as plt
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SequentialChain 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

file_name= 'combined2.0.csv'
# Set up navbar
st.set_page_config(page_title='Properlytics', page_icon=':house:', layout='wide')
menu = ['Home', 'Analytics', 'About']
choice = st.sidebar.selectbox('Select a page', menu)

# App framework
if choice == 'Analytics':
    @st.cache_data 
    def load_data(nrows):
        data = pd.read_csv(file_name, nrows=nrows)
        return data
    
    #load data
    data = load_data(8507)

    st.title('Data Analisis of the Real Estate Market in Turin') 
    st.write('Here you can find some data analisis about the real estate market in Turin')

    #setting colors to plot
    colorscale = ["red", "blue"]

    col1, col2 = st.columns(2)
    with col1:
        # Crea il DataFrame con le colonne specificate
        df = pd.DataFrame(data, columns=['Dimension (m2)', 'Price'])
        # Crea il plot
        st.subheader('Comparison between the price and the dimension of the houses in sale')
        # Crea lo scatter plot utilizzando Plotly Express
        fig = px.scatter(
            df, 
            x='Dimension (m2)', 
            y='Price',
            title="<b>Price-Dimension</b>",
            color=df['Price'],
            color_continuous_scale=colorscale,
            labels={'x': 'Zone', 'y': 'House in sale'}
        )
        fig.update_layout(xaxis_title='Dimension (m2)', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader('Heat map of the houses in sale in Turin')
        #add a space
        st.write("")
        st.write("")
        df = pd.read_csv('combined_with_coords.csv', on_bad_lines='skip')
        # Creazione della mappa di densità
        fig = px.density_mapbox(
            df,
            lat="latitude",
            lon="longitude",
            z="Price",
            radius=20,
            center={"lat": 45.0705, "lon": 7.6868},
            zoom=10,
            mapbox_style="open-street-map",
            title="Heat-map based on Price",
            color_continuous_scale="Jet",
            opacity=0.6,
        )

        # Mostra la mappa
        st.plotly_chart(fig, use_container_width=True)
    
    
    # Crea il DataFrame con le colonne specificate
    df = pd.DataFrame(data, columns=['Zona', 'Price'])
    # Crea il plot
    st.subheader('Insights of the number of houses in sale for each area')
    # Calcola il numero di case in vendita per ogni zona
    zone_counts = df['Zona'].value_counts().sort_values(ascending=True)
    # Crea il plot utilizzando Plotly Express
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            x=zone_counts.values, 
            y=zone_counts.index, 
            title="<b>All the properties</b>",
            color=zone_counts.values,
            color_continuous_scale=colorscale,
            labels={'x': 'Zone', 'y': 'House in sale'}
        )
        
        fig.update_layout(
            xaxis_title="Number of houses in sale",  # Etichetta dell'asse x 
            yaxis_title="Zone",  # Etichetta dell'asse y
        )

        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # ---- SIDEBAR ----
        st.sidebar.header("Tune analysis 🕹 ")
        default_zona = ["Santa Rita"]
        zona = st.sidebar.multiselect(
            "Select the Neighborhood for the number of properties:",
            options=data["Zona"].unique(),
            default=default_zona,
        )
        #--- END SIDEBAR ---
        # Crea il DataFrame con le colonne specificate
        df = pd.DataFrame(data, columns=['Zona', 'Price'])
        data_selection = data.query("Zona == @zona")
        number = (data_selection['Zona'].value_counts().sort_values(ascending=True))
        fig_select = px.bar(
            number,
            x=number.values,  # Utilizza i valori come etichette sull'asse x
            y=number.index, 
            orientation="h",
            title="<b>Select the zone to see the number properties for each Neighbor</b>",
            color=number.values,
            color_continuous_scale=colorscale,
            template="plotly_dark"
        )

        fig_select.update_layout(
            xaxis_title="Number of houses in sale",  # Etichetta dell'asse x
            yaxis_title="Zone",  # Etichetta dell'asse y
        )

        st.plotly_chart(fig_select, use_container_width=True)

    
   
    # Carica i dati dal file CSV
    df = pd.read_csv('combined_with_coords.csv', on_bad_lines='skip')
    st.subheader('Heat map of the houses in sale in Turin for each zone')
    # Creazione della barra laterale per la selezione della zona
    default_zona = ["Santa Rita"]
    key = "zona_selection"
    zona = st.sidebar.multiselect(
        "Select the Zone to see the Heat-Map:",
        options=df["Zona"].unique(),
        default=default_zona,
        key=key
    )

    # Filtra i dati in base alle zone selezionate
    data_selection = df[df["Zona"].isin(zona)]

    # Controlla se sono state selezionate delle zone
    if not data_selection.empty:
        # Crea il grafico di densità
        fig = px.density_mapbox(
            data_selection,
            lat="latitude",
            lon="longitude",
            z="Price",
            radius=20,
            center={"lat": 45.0705, "lon": 7.6868},
            zoom=10,
            mapbox_style="open-street-map",
            title="Turin",
            color_continuous_scale="Jet",
            opacity=0.6,
        )

        # Mostra il grafico
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No data selected")

    st.subheader('Insights of all the houses in sale')
    # Carica i dati dal file CSV
    df = pd.read_csv('combined_with_coords.csv', on_bad_lines='skip')
    # Converti le colonne "latitude" e "longitude" in numeri
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    # Filtra le righe con valori numerici validi per latitudine e longitudine
    df = df.dropna(subset=['latitude', 'longitude'])
    # Mostra la mappa dei dati
    st.map(df)
# Add your FAQ content here
elif choice == 'About':
    st.title('About Properlytics')
    st.write('Properlytics is a real estate analytics platform that helps you make smarter real estate decisions through predictive analytics. We are a team of data scientists, engineers, and real estate professionals who are passionate about helping you make better real estate decisions.')