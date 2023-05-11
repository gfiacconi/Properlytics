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
if choice == 'Home':
    st.title('🏘 📈Properlytics: Smarter Real Estate Decisions through Predictive Analytics')

    col1, col2= st.columns(2)
    col1.metric("Prezzo medio vendita (€/m²)", "€ 1.923", "+ 2,34%")
    col2.metric("Prezzo medio affitto (€/m²)", "€ 10,12", "+ 10,84%")

    col1, col2= st.columns(2)

    with col1:
        squareMeter = st.slider('insert sqaure meter', 0, 800, 100)
        st.write('You selected:', squareMeter, 'm²')

    with col2:
        option = st.selectbox(
        'Floor',
        ('1', '2', '3','4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14','15', '16', '17', '18', '19', '20', '21', '22', '23', '24'))


    genre = st.radio(
    "enter the area",
    ('Centro', 'Borgo Po', 'San Salvario', 'Crocetta'))

    prompt = f"square meter: {squareMeter} \n floor: {option} \n area: {genre}"
    # Llms
    llm = OpenAI(temperature=0.1) 

    #wiki = WikipediaAPIWrapper()

    reader = PdfReader('dataTorino.pdf')

    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(        
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()

    docsearch = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # Show stuff to the screen if there's a prompt
    if st.button('submit'):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.1)
            my_bar.progress(percent_complete + 1, text=progress_text)
        response = llm(prompt)
        docs = docsearch.similarity_search(prompt)
        st.write(chain.run(input_documents=docs, question='parla in un modo articolato da venditore ad ogni cosa che devi rispondere' + prompt + 'descrivi la zona e di qualsiasi cosa di interessante'))
        col2.metric("Final Price",chain.run(input_documents=docs, question='in base a questi dati:' + prompt + 'calcola il prezzo finale in base ai metri quadri inseriti, scrivi solo il prezzo finale senza nient altro e senza spazi o punti'))

elif choice == 'Analytics':
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

        # Configurazione Streamlit
        st.title('Plot of the price and dimension of the houses in sale')
        # Crea il plot
        st.subheader('Plot')

        # Crea lo scatter plot utilizzando Plotly Express
        fig = px.scatter(
            df, 
            x='Dimension (m2)', 
            y='Price',
            title="<b>All the properties</b>",
            color=df['Price'],
            color_continuous_scale=colorscale,
            labels={'x': 'Zone', 'y': 'House in sale'}
        )
        fig.update_layout(xaxis_title='Dimension (m2)', yaxis_title='Price')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
                # Crea il DataFrame con le colonne specificate
        df = pd.DataFrame(data, columns=['Rooms', 'Bathrooms'])
        zone_counts = df['Rooms'].value_counts().sort_values(ascending=True)

        # Configurazione Streamlit
        st.title('Plot of the price and dimension of the houses in sale')
        # Crea il plot
        st.subheader('Plot')
        # Crea lo scatter plot utilizzando Plotly Express
        fig = px.bar(
            df, 
            x=zone_counts.values, 
            y=zone_counts.index,
            title="<b>All the properties</b>",
            color=zone_counts.values,
            color_continuous_scale=colorscale
        )
        fig.update_layout(xaxis_title='Bathrooms', yaxis_title='Rooms')
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
            yaxis_title="Zona",  # Etichetta dell'asse y
        )

        st.plotly_chart(fig, use_container_width=True)
    with col2:
        # ---- SIDEBAR ----
        st.sidebar.header("Please Filter Here:")
        default_zona = ["Santa Rita"]
        zona = st.sidebar.multiselect(
            "Select the Type:",
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
            title="<b>Select the zone to see the properties</b>",
            color=number.values,
            color_continuous_scale=colorscale,
            template="plotly_dark"
        )

        fig_select.update_layout(
            xaxis_title="Numero di Proprietà",  # Etichetta dell'asse x
            yaxis_title="Zona",  # Etichetta dell'asse y
        )

        st.plotly_chart(fig_select, use_container_width=True)

    # Carica i dati dal file CSV
    df = pd.read_csv('combined_with_coords.csv', on_bad_lines='skip')

    # Converti le colonne "latitude" e "longitude" in numeri
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    # Filtra le righe con valori numerici validi per latitudine e longitudine
    df = df.dropna(subset=['latitude', 'longitude', 'Price'])
    # Mostra la mappa dei dati
    st.map(df)

# Add your FAQ content here
elif choice == 'About':
    st.title('About Properlytics')
    st.write('Properlytics is a real estate analytics platform that helps you make smarter real estate decisions through predictive analytics. We are a team of data scientists, engineers, and real estate professionals who are passionate about helping you make better real estate decisions.')