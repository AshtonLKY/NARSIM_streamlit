"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""
import xml.etree.ElementTree as ET
import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import re
import json
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import List, Tuple, Dict, Any, Optional

# Streamlit page configuration
st.set_page_config(
    page_title="Ollama PDF RAG Streamlit UI",
    page_icon="🎈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner=True)
def extract_model_names(
    models_info: Dict[str, List[Dict[str, Any]]],
) -> Tuple[str, ...]:
    """
    Extract model names from the provided models information.

    Args:
        models_info (Dict[str, List[Dict[str, Any]]]): Dictionary containing information about available models.

    Returns:
        Tuple[str, ...]: A tuple of model names.
    """
    logger.info("Extracting model names from models_info")
    model_names = tuple(model["name"] for model in models_info["models"])
    logger.info(f"Extracted model names: {model_names}")
    return model_names


def create_vector_db(file_upload) -> Chroma:
    """
    Create a vector database from an uploaded PDF file.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        Chroma: A vector store containing the processed document chunks.
    """
    logger.info(f"Creating vector DB from file upload: {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)
    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to temporary path: {path}")
        loader = UnstructuredPDFLoader(path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=550, chunk_overlap=200)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="myRAG"
    )
    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


def process_question(question: str, vector_db: Chroma, selected_model: str) -> str:
    """
    Process a user question using the vector database and selected language model.

    Args:
        question (str): The user's question.
        vector_db (Chroma): The vector database containing document embeddings.
        selected_model (str): The name of the selected language model.

    Returns:
        str: The generated response to the user's question.
    """
    logger.info(f"""Processing question: {
                question} using model: {selected_model}""")
    # json_schema = {"aircraft 1": {
    #     "departure": {
    #       "af": "ICAO code of departure airfield here"
    #     },
    #     "initial_position": {
    #       "latitude": "latitude in DMS format for example 023106.70N",
    #       "longitude": "latitude in DMS format for example 1035709.81E",
    #       "altitude": "Altitude reading for example FL160",
    #       "heading": "heading in degrees 32.053335700277444"
    #     },
    #     "air_route": """list of waypoints that make up the air route, for example ["RAXIM", "OTLON", "VISAT", "DUBSA", "DAMOG", "DOLOX", "DUDIS"]""",
    #     "destination": {
    #       "af": "ICAO code of destination airfield here"
    #     },
    #     "time": "starting time of aircraft initialization in the simulator as an integer representing seconds",
    #     "type": "type of aircraft, for example A320"}}
    # dumps = json.dumps(json_schema, indent=2)
    llm = ChatOllama(model=selected_model,
    #format = 'json',
    temperature=0.1)
  
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        # template="""You are an AI language model assistant. Your task is to generate 3
        # different versions of the given user question to retrieve relevant documents from
        # a vector database. By generating multiple perspectives on the user question, your
        # goal is to help the user overcome some of the limitations of the distance-based
        # similarity search. Provide these alternative questions separated by newlines.
        # Original question: {question}""",
         template="""You are an AI language model assistant. Your task is to generate 2
        different versions of the given user question to retrieve relevant documents from
        a vector database. By generating multiple perspectives on the user question, your
        goal is to help the user overcome some of the limitations of the distance-based
        similarity search. Provide these alternative questions separated by newlines.
        Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT, include_original=True
    )
    retrieved_docs = retriever.get_relevant_documents(query=QUERY_PROMPT)
    st.write(retrieved_docs)
    # results = vector_db.similarity_search_with_score(question, k=3)

    # retriever = "\n\n---\n\n".join(
    #     [doc.page_content for doc, _score in results])

    template = """Answer or perform the user requested task with help from the following context:
    {context}
    User Request: {question}
    If the user requests to generate aircraft scenarios please provide the answer in the form of JSON in the following schema:
    {{"aircraft 0": {{
        "departure": {{
          "af": "ICAO code of departure airfield here"
        }},
        "initial_position": {{
          "latitude": "latitude in DMS format for example 023106.70N",
          "longitude": "latitude in DMS format for example 1035709.81E",
          "altitude": "Altitude reading for example FL160",
          "heading": "heading in degrees 32.053335700277444"
        }},
        "air_route": "list of waypoints that make up the air route, for example ["RAXIM", "OTLON", "VISAT", "DUBSA", "DAMOG", "DOLOX", "DUDIS"]",
        "destination": {{
          "af": "ICAO code of destination airfield here"
        }},
        "time": "starting time of aircraft initialization in the simulator as an integer representing seconds",
        "type": "type of aircraft, for example A320"}}

    If there are multiple aircraft, put all aircraft in a single dictionary with the keys labelled aircraft 1, aircraft 2, aircraft 3, etc.
    If the user does not request for a scenario, simply reply normally, do not ever give Python code.
    """

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(question)
    
    logger.info("Question processed and response generated")
    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    """
    Extract all pages from a PDF file as images.

    Args:
        file_upload (st.UploadedFile): Streamlit file upload object containing the PDF.

    Returns:
        List[Any]: A list of image objects representing each page of the PDF.
    """
    logger.info(f"""Extracting all pages as images from file: {
                file_upload.name}""")
    pdf_pages = []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages = [page.to_image().original for page in pdf.pages]
    logger.info("PDF pages extracted as images")
    return pdf_pages


def delete_vector_db(vector_db: Optional[Chroma]) -> None:
    """
    Delete the vector database and clear related session state.

    Args:
        vector_db (Optional[Chroma]): The vector database to be deleted.
    """
    logger.info("Deleting vector DB")
    if vector_db is not None:
        vector_db.delete_collection()
        st.session_state.pop("pdf_pages", None)
        st.session_state.pop("file_upload", None)
        st.session_state.pop("vector_db", None)
        st.success("Collection and temporary files deleted successfully.")
        logger.info("Vector DB and related session state cleared")
        st.rerun()
    else:
        st.error("No vector database found to delete.")
        logger.warning("Attempted to delete vector DB, but none was found")


def str2dict(response):
    """
    Process each chunk of the streamed response.
    """
    # Define the regular expression pattern
    pattern = r'{.*}'

    # Find all matches
    matches = re.findall(pattern, response, re.DOTALL)
    json_str = ' '.join(matches)
    
    try:
        json_d = json.loads(json_str)
        # st.write(json_d)
    except json.JSONDecodeError:
        json_d = ""
    return json_d


# def flatten_json(data):

#     if not data:
#         print("JSON string is empty!")
#         return None

#     if isinstance(data, dict):
#         # Check if the dictionary has the required keys
#         if all(key in data for key in ["departure", "initial_position", "air_route", "destination", "time", "type"]):
#             return data

#         # Recursively check each value in the dictionary
#         for key in data:
#             extracted = flatten_json(data[key])
#             if extracted:
#                 return extracted

#     return None


def json_to_xml(json_data):
  xml_list = []
  st.write(json_data)
  for i, item in enumerate(json_data.values()):
    root = ET.Element("initial-flightplans", key="initial-flightplans: "+str(i))

    usage = ET.SubElement(root, "usage")
    usage.text = "ALL"
    time = ET.SubElement(root, "time")
    time.text = str(item["time"])
    callsign = ET.SubElement(root, "callsign")
    callsign.text = "SQ123"
    rules = ET.SubElement(root, "rules")
    squawk = ET.SubElement(root, "squawk", units="octal")
    squawk.text = "0000"
    aircraft_type = ET.SubElement(root, "type")
    aircraft_type.text = item["type"]
    waketurb = ET.SubElement(root, "waketurb")
    waketurb.text = "MEDIUM"
    equip = ET.SubElement(root, "equip")
    vehicle_type = ET.SubElement(root, "vehicle_type")

    dep = ET.SubElement(root, "dep")
    dep_af = ET.SubElement(dep, "af")
    dep_af.text = item["departure"]["af"]
    dep_rwy = ET.SubElement(dep, "rwy")

    des = ET.SubElement(root, "des")
    des_af = ET.SubElement(des, "af")
    des_af.text = item["destination"]["af"]

    for route in item["air_route"]:
        air_route = ET.SubElement(root, "air_route")
        air_route.text = route

    rfl = ET.SubElement(root, "rfl")
    rfl.text = "ALL"
    init = ET.SubElement(root, "init")
    pos = ET.SubElement(init, "pos")
    lat = ET.SubElement(pos, "lat")
    lat.text = item["initial_position"]["latitude"]
    lon = ET.SubElement(pos, "lon")
    lon.text = item["initial_position"]["longitude"]

    freq = ET.SubElement(init, "freq")
    alt = ET.SubElement(init, "alt", units="")
    alt.text = item["initial_position"]["altitude"]
    hdg = ET.SubElement(init, "hdg")
    hdg.text = item["initial_position"]["heading"]
    xml_list.append(ET.tostring(root, encoding='unicode'))

  return xml_list


def main() -> None:
    """
    Main function to run the Streamlit application.

    This function sets up the user interface, handles file uploads,
    processes user queries, and displays results.
    """
    st.subheader("🧠 Ollama PDF RAG playground", divider="gray", anchor=False)

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )

    file_upload = col1.file_uploader(
        "Upload a PDF file ↓", type="pdf", accept_multiple_files=False
    )

    if file_upload:
        st.session_state["file_upload"] = file_upload
        if st.session_state["vector_db"] is None:
            st.session_state["vector_db"] = create_vector_db(file_upload)
        pdf_pages = extract_all_pages_as_images(file_upload)
        st.session_state["pdf_pages"] = pdf_pages

        zoom_level = col1.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50
        )

        with col1:
            with st.container(height=410, border=True):
                for page_image in pdf_pages:
                    st.image(page_image, width=zoom_level)

    delete_collection = col1.button("⚠️ Delete collection", type="secondary")

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    with col2:
        message_container = st.container(height=500, border=True)

        for message in st.session_state["messages"]:
            avatar = "🤖" if message["role"] == "assistant" else "😎"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append(
                    {"role": "user", "content": prompt})
                message_container.chat_message(
                    "user", avatar="😎").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first.")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role": "assistant", "content": response}
                    )
                json_d = str2dict(response)

                
                if json_d:  
                    # print(f"Valid JSON string: {json_str}")
                    st.write([json_to_xml(json_d)])
                else:
                    st.error("The ouput does not contain valid JSON.")
            except Exception as e:
                st.error(e, icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to begin chat...")


if __name__ == "__main__":
    main()
