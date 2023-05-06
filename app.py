import os 
import streamlit as st 
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.utilities import WikipediaAPIWrapper
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain, SequentialChain 
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS

os.environ["OPENAI_API_KEY"] == st.secrets['openai']["OPENAI_API_KEY"]

# Set up navbar
st.set_page_config(page_title='Properlytics', page_icon=':house:', layout='wide')
menu = ['Home', 'FAQ', 'About']
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
        response = llm(prompt)
        #wiki_research = wiki.run(prompt) 
        docs = docsearch.similarity_search(prompt)
        st.write(chain.run(input_documents=docs, question='parla in un modo articolato da venditore ad ogni cosa che devi rispondere' + prompt + 'descrivi la zona e di qualsiasi cosa di interessante'))
        #st.write(chain.run(input_documents=docs, question='in base a questi dati:' + prompt + 'calcola il prezzo finale in base ai metri quadri inseriti, scrivi solo il prezzo finale senza nient altro'))
        #col2.metric("Final Price",chain.run(input_documents=docs, question='in base a questi dati:' + prompt + 'calcola il prezzo finale in base ai metri quadri inseriti, scrivi solo il prezzo finale senza nient altro e senza spazi o punti'))
        st.write(chain.run(input_documents=docs, question='in base al valore di'+ f"{genre}" +"quanto vale?"))

        st.write(chain.run(input_documents=docs, question='ripeti questo numero: ' + f"{squareMeter}"))
        #st.write(wiki_research)
        #st.write(response)

elif choice == 'FAQ':
    st.title('Frequently Asked Questions')
    # Add your FAQ content here
elif choice == 'About':
    st.title('About Properlytics')
    # Add your About content here
