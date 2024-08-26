"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""
import xml.etree.ElementTree as ET
import xml.dom.minidom
import streamlit as st
import logging
import os
import tempfile
import shutil
import pdfplumber
import ollama
import re
import json
import random
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_cohere import ChatCohere
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Tuple, Dict, Any, Optional


# Streamlit page configuration
st.set_page_config(
    page_title="Ollama Conversational RAG Streamlit UI",
    page_icon="✈️",
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

    text_splitter = RecursiveCharacterTextSplitter(separators=["-----------", "\n\n", "\n"],
                                                   chunk_size=550, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    logger.info("Document split into chunks")

    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large", show_progress=True)
    vector_db = Chroma.from_documents(
        documents=chunks, embedding=embeddings, collection_name="myRAG"
    )
    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir} removed")
    return vector_db


def get_history(chat):  # added get_history function
    chat_history = []
    # recall_value = 6
    # for m in chat:
    #     if m["role"] == 'system':
    #         chat_history.append(SystemMessage(content=m["content"]))
    #     elif m['role'] == 'user':
    #         chat_history.append(HumanMessage(content=m["content"]))
    #     elif m["role"] == "assistant":
    #         chat_history.append(AIMessage(content=m["content"]))
    for m in chat:
        if m['role'] == 'user':
            chat_history.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            chat_history.append(AIMessage(content=m["content"]))
    return chat_history


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
    llm = ChatCohere(cohere_api_key='njPNUMUWPRMIoHRPoV8xSJvucU2sEZw99puyga7r',
                     model="command-r", temperature=0, streaming=True,
                     preamble="""You are an AI assistant capable of generating air traffic scenarios for a simulator.
                    - When the user requests for a scenario, strictly give the output in JSON, otherwise reply normally.
                    - Do not give python code.
                    - Refer to the chat_history to understand the context of the conversation before replying.
                    - Use complete air_routes from the Data Bundle when generating outputs.
                    - Ensure that the aircraft are separated by a minimum time 100 seconds.

                    ---------------------------------------------
                    Only use the Available Aircraft Types:
                    • A320
                    • B737
                    • B738
                    • B734
                    • B744
                    • A388
                    • A333
                   -----------------------------------------------
                   If the number of aircraft are not specified, use the following range:
                   • Low density traffic : 1 to 4 aircraft.
                   • Medium density traffic: 5 to 9 aircraft.
                   • High density traffic: 10 to 20 aircraft.
                   Otherwise, ensure to generate the same number of aircraft that are specified.""")
    # llm = ChatOllama(model="selected_model",
    #                  # format = 'json',
    #                  temperature=0)

    # QUERY_PROMPT = PromptTemplate(
    #     input_variables=["question"],
    #     template="""You are an AI language model assistant. Your task is to generate 3
    #     different versions of the given user question to retrieve relevant documents from
    #     a vector database. By generating multiple perspectives on the user question, your
    #     goal is to help the user overcome some of the limitations of the distance-based
    #     similarity search. Original question: {question}""",
    # )

    # retriever = MultiQueryRetriever.from_llm(
    #     vector_db.as_retriever(), llm=llm, include_original=True
    # )
    retriever = vector_db.as_retriever(
        search_type="mmr", search_kwargs={"fetch_k": 25, "k": 16, "lambda_mult": 0.5})

    extracted_docs = retriever.invoke(question)

    def format_docs(docs):
        # st.write("\n\n".join(doc.page_content for doc in docs))
        # , "\n\n".join(doc.page_content for doc in docs[:3])
        return "\n\n".join(doc.page_content for doc in docs)
    # formatted_docs, extracted_context = format_docs(extracted_docs)
    formatted_docs = format_docs(extracted_docs)
    st.write([formatted_docs])
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             """You are an AI agent that answers or performs the user requested task with help from the context provided by the vector database. If the requested, use information from the vector database ONLY.
    #         Context from vector database: {context}

    #         ---------------------------------------------------

    #         If the user requests to generate aircraft scenarios please provide the answer in the form of JSON in the following schema:
    #         {{"aircraft 0": {{
    #             "departure": {{
    #               "af": "ICAO code of departure airfield here"
    #             }},
    #             "initial_position": {{
    #               "latitude": "latitude in DMS format for example 023106.70N",
    #               "longitude": "latitude in DMS format for example 1035709.81E",
    #               "altitude": "Altitude reading for example FL160",
    #               "heading": "heading in degrees 32.053335700277444"
    #             }},
    #             "air_route": "list of waypoints that make up the air route, for example ["RAXIM", "OTLON", "VISAT", "DUBSA", "DAMOG", "DOLOX", "DUDIS"]",
    #             "destination": {{
    #               "af": "ICAO code of destination airfield here"
    #             }},
    #             "time": "starting time of aircraft initialization in the simulator as an integer representing seconds",
    #             "type": "type of aircraft, for example A320"}}

    #         If there are multiple aircraft, put all aircraft in a single dictionary with the keys labelled aircraft 1, aircraft 2, aircraft 3, etc.
    #         You are to use flight information found in the vector database only.
    #         If the user does not request for a scenario, simply reply normally, do not ever give Python code. Look into the chat_history to understand the context of the conversation before answering.
    #       """),
    #         MessagesPlaceholder(variable_name="chat_history"),
    #         ("human", "User Prompt: {question}"),
    #     ]
    # )
    prompt = ChatPromptTemplate.from_messages(
        [
            # (
            #     "system",
            #     """You are an AI assistant capable of generating air traffic scenarios for a simulator.
            #         - When the user requests for a scenario, strictly give the output in JSON, otherwise reply normally.
            #         - Do not give python code.
            #         - Refer to the chat_history to understand the context of the conversation before replying.
            #         - Use complete air_routes when generating outputs.

            #         ---------------------------------------------

            #         The Data bundle provides examples of flight paths, please refer to them and use air routes from the examples when creating scenarios
            #         Data bundle: {context}

            #         ---------------------------------------------
            #         Only use the Available Aircraft Types:
            #         • A320
            #         • B737
            #         • B738
            #         • B734
            #         • B744
            #         • A388
            #         • A333
            #        -----------------------------------------------
            #        If the number of aircraft are not specified, use the following range:
            #        • Low density traffic : 1 to 4 aircraft.
            #        • Medium density traffic: 5 to 9 aircraft.
            #        • High density traffic: 10 to 20 aircraft.

            #        - Ensure that the aircraft are separated by a minimum time 100 seconds. Generate these aircraft on different air_routes.

            #        Otherwise, ensure to generate the same number of aircraft that are specified.
            #   """),
            MessagesPlaceholder(variable_name="chat_history"),

            ("human", """User prompt: {question}
                        ----------------------------------------
                        If the user requests for scenario generation, please use the following schema_1, otherwise reply normally:
                        {{"aircraft 0": {{
                        "departure": {{
                        "af": "ICAO code of departure airfield here"
                        }},
                        "initial_position": {{
                        "latitude": "latitude in DMS format for example 023106.70N",
                        "longitude": "latitude in DMS format for example 1035709.81E",
                        "altitude": "Altitude reading for example FL300",
                        "heading": "heading in degrees for example 32.053335700277444"
                        }},
                        "air_route": "list of waypoints that make up the air route, for example ["RAXIM", "OTLON", "VISAT", "DUBSA", "DAMOG", "DOLOX", "DUDIS"]",
                        "destination": {{
                        "af": "ICAO code of destination airfield here"
                        }},
                        "time": "starting time of aircraft initialization in the simulator as an integer representing seconds, for example 300",
                        "type": "type of aircraft, for example A320"}}
                        --------------------------------------------
                        The Data bundle provides examples of airways, please refer to them and use air routes from the examples when creating scenarios
                        Data bundle: {context}
                       """),
        ]
    )
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         (
    #             "system",
    #             """You are an AI assistant capable of generating air traffic scenarios for a simulator. The simulator requires  6 parameters shown in schema_1. If multiple aircraft are requested, label them as "aircraft 0", "aircraft 1", and so on.

    #             schema_1:
    #              {{"aircraft 0": {{
    #             "time": "starting time of aircraft initialization in the simulator as an integer representing seconds",
    #             "type": "type of aircraft, for example A320",
    #             "departure": {{
    #               "af": "ICAO code of departure airfield here"
    #             }},
    #             "initial_position": {{
    #               "latitude": "latitude in DMS format for example 023106.70N",
    #               "longitude": "latitude in DMS format for example 1035709.81E",
    #               "altitude": "Altitude reading for example FL160",
    #               "heading": "heading in degrees 32.053335700277444"
    #             }},
    #             "air_route": "list of waypoints that make up the air route, for example ["RAXIM", "OTLON", "VISAT", "DUBSA", "DAMOG", "DOLOX", "DUDIS"]",
    #             "destination": {{
    #               "af": "ICAO code of destination airfield here"
    #             }}
    #             }}

    #             Below is the explanation of the 6 parameters in schema_1:
    #             Based on the provided PDF document, here is the explanation of each key in the given JSON text:

    #             1. **"departure"**:
    #             - **"af"**: Represents the ICAO (International Civil Aviation Organization) code of the departure airfield. It should be a string of maximum size 4 characters.

    #             2. **"initial_position"**:
    #             - **"latitude"**: Latitude in Degrees, Minutes, and Seconds (DMS) format, e.g., 023106.70N.
    #             - **"longitude"**: Longitude in DMS format, e.g., 1035709.81E.
    #             - **"altitude"**: Altitude reading in meters above sea level, e.g., FL160, where FL represents flight level.
    #             - **"heading"**: Heading in degrees (0-360), e.g., 32.053335700277444.

    #             The initial_position represents the position the aircraft will initialize to once the simulation starts.

    #             3. **"air_route"**: List of waypoints that make up the air route. Each waypoint is a string with a maximum size of 10 characters. For example, ["RAXIM", "OTLON", "VISAT", "DUBSA", "DAMOG", "DOLOX", "DUDIS"].

    #             The simulated aircraft will travel from the initial_position along these air routes.

    #             4. **"destination"**:
    #             - **"af"**: Represents the ICAO code of the destination airfield. It should be a string of maximum size 4 characters.

    #             5. **"time"**: Starting time of aircraft initialization in the simulator, represented as an integer in seconds.

    #             6. **"type"**: Type of aircraft, e.g., A320. This should be a string with a maximum size of 4 characters.

    #             When generating scenarios, consider the common air routes from the data bundle and find their respective coordinates, which are also found in the data bundle. Then knowing the path and distance the aircraft is going to take via calculation from the series of coordinates, set an initialization position and time such that the scenario will play out in accordance to the user's request.
    #             Data bundle: {context}

    #             When the user requests for a scenario, strictly give the output in JSON based on schema_1. Do not give python code. If the user asks for something unrelated, just reply normally. Refer to the chat_history to understand the context of the conversation before replying.

    #       """),
    #         MessagesPlaceholder(variable_name="chat_history"),
    #         ("human", "{question}"),
    #     ]
    # )
    # extracted_docs = retriever.get_relevant_documents(query=QUERY_PROMPT)
    # st.write(extracted_docs)
    chat_history = get_history(st.session_state.messages)
    st.write(chat_history)
    chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(
        {"context": formatted_docs, "chat_history": chat_history, "question": question})
    # extracted_context = """Context provided: """ + extracted_context #comment it out if you do not wish to keep context in chat history

    logger.info("Question processed and response generated")
    return response  # , extracted_context#comment it out if you do not wish to keep context in chat history


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
    waket = {'A320': 'MEDIUM', 'B738': 'MEDIUM', 'B737': 'MEDIUM', 'B744': 'HEAVY',
             'B734': 'MEDIUM', 'A388': 'HEAVY', 'A333': 'HEAVY'}
    # Create the root element <ifp>
    root_ifp = ET.Element("ifp")

    # Add experiment-date element
    experiment_date = ET.SubElement(root_ifp, "experiment-date")
    day = ET.SubElement(experiment_date, "day")
    day.text = '6'
    month = ET.SubElement(experiment_date, "month")
    month.text = '3'
    year = ET.SubElement(experiment_date, "year")
    year.text = '2022'

    # Add experiment-time element
    experiment_time = ET.SubElement(root_ifp, "experiment-time")
    experiment_time.text = '00000'

    # Add default-equipment element
    default_equipment = ET.SubElement(root_ifp, "default-equipment")
    default_equipment.text = "SSR_MODE_A+SSR_MODE_C+P_RNAV+FMS+FMS_GUIDANCE_VNAV+FMS_GUIDANCE_SPEED"

    for i, item in enumerate(json_data.values()):
        if isinstance(item, str) and " " in item:
            # Remove all spaces in the string
            item = item.replace(" ", "")
        # Create the initial-flightplans element
        initial_flightplans = ET.SubElement(
            root_ifp, "initial-flightplans", key="initial-flightplans: "+str(i))

        usage = ET.SubElement(initial_flightplans, "usage")
        usage.text = "ALL"
        time = ET.SubElement(initial_flightplans, "time")
        time.text = str(item["time"])  # From model
        callsign = ET.SubElement(initial_flightplans, "callsign")
        callsign.text = "SQ1"+str(i)
        rules = ET.SubElement(initial_flightplans, "rules")
        squawk = ET.SubElement(initial_flightplans, "squawk", units="octal")
        squawk.text = str(random.randint(1000, 9999))
        aircraft_type = ET.SubElement(initial_flightplans, "type")
        aircraft_type.text = item["type"]  # From model
        waketurb = ET.SubElement(initial_flightplans, "waketurb")
        waketurb.text = waket[item["type"]]
        equip = ET.SubElement(initial_flightplans, "equip")
        vehicle_type = ET.SubElement(initial_flightplans, "vehicle_type")

        dep = ET.SubElement(initial_flightplans, "dep")
        dep_af = ET.SubElement(dep, "af")
        dep_af.text = item["departure"]["af"]  # From model
        dep_rwy = ET.SubElement(dep, "rwy")

        des = ET.SubElement(initial_flightplans, "des")
        des_af = ET.SubElement(des, "af")
        des_af.text = item["destination"]["af"]  # From model

        for route in item["air_route"]:
            air_route = ET.SubElement(initial_flightplans, "air_route")
            air_route.text = route

        rfl = ET.SubElement(initial_flightplans, "rfl")
        rfl.text = '300'
        init = ET.SubElement(initial_flightplans, "init")
        pos = ET.SubElement(init, "pos")
        lat = ET.SubElement(pos, "lat")
        lat.text = item["initial_position"]["latitude"]  # From model
        lon = ET.SubElement(pos, "lon")
        lon.text = item["initial_position"]["longitude"]  # From model

        freq = ET.SubElement(init, "freq")
        freq.text = 'SINRADS1'
        alt = ET.SubElement(
            init, "alt", units=item["initial_position"]["altitude"][:2])
        alt.text = item["initial_position"]["altitude"][2:]  # From model
        hdg = ET.SubElement(init, "hdg")
        hdg.text = item["initial_position"]["heading"]

    # Convert the tree to a string
    xdat_content = ET.tostring(
        root_ifp, encoding='UTF-8', xml_declaration=True)
    # print(xdat_content)
    dom = xml.dom.minidom.parseString(xdat_content)
    pretty_xdat = dom.toprettyxml()

    return pretty_xdat


def main() -> None:
    """
    Main function to run the Streamlit application.

    This function sets up the user interface, handles file uploads,
    processes user queries, and displays results.
    """
    st.subheader("✈️ NARSIM Scenario Generator Assistant",
                 divider="gray", anchor=False)

    models_info = ollama.list()
    available_models = True  # extract_model_names(models_info)

    col1, col2 = st.columns([1.5, 2])

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    if available_models:
        selected_model = col2.selectbox(
            # available_models
            "Pick a model available locally on your system ↓",
            ["Command-R-35b"]
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
            # if message['role'] != "system":
            avatar = "🤖" if message["role"] == "assistant" else "👨‍✈️"
            with message_container.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])

        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append(
                    {"role": "user", "content": prompt})
                message_container.chat_message(
                    "user", avatar="👨‍✈️").markdown(prompt)

                with message_container.chat_message("assistant", avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            # response, extracted_context = process_question(
                            #     prompt, st.session_state["vector_db"], selected_model
                            # )
                            response = process_question(
                                prompt, st.session_state["vector_db"], selected_model
                            )
                            # print(extracted_context)
                            # st.session_state["messages"].append(
                            #     {"role": "system", "content": extracted_context}
                            # )
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
                    # st.write([json_to_xml(json_d)])

                    # Generate XDAT content
                    pretty_xdat = json_to_xml(json_d)

                    # Display XDAT content
                    st.write(pretty_xdat)
                    st.download_button(
                        label='Download .xdat', data=pretty_xdat, file_name="downloaded_scenario.xdat")
                    # Button to download XDAT file

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
