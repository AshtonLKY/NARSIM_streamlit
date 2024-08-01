import xml.etree.ElementTree as ET
import re
import json
import ollama
import streamlit as st
from openai import OpenAI
from utilities.icon import page_icon

st.set_page_config(
    page_title="Chat playground",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def extract_model_names(models_info: list) -> tuple:
    """
    Extracts the model names from the models information.

    :param models_info: A dictionary containing the models' information.

    Return:
        A tuple containing the model names.
    """

    return tuple(model["name"] for model in models_info["models"])


# def extract_json_string(input_string):
#     # Regular expression to find JSON in a string
#     json_pattern = re.compile(r'(\{(?:[^{}]|(?R))*\})')
#     match = json_pattern.search(input_string)

#     if match:
#         json_candidate = match.group()
#         try:
#             # Try to load the extracted substring as JSON
#             json_data = json.loads(json_candidate)
#             # If successful, assign the extracted substring to json_str
#             json_str = json_candidate
#             return json_str
#         except json.JSONDecodeError:
#             # If a JSONDecodeError is raised, the substring is not valid JSON
#             return None
#     else:
#         return None

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
    except json.JSONDecodeError:
        print("Can't convert to JSON, check string")
    return json_d


def flatten_json(data):

    if not data:
        print("JSON string is empty!")
        return None

    if isinstance(data, dict):
        # Check if the dictionary has the required keys
        if all(key in data for key in ["departure", "time", "type", "initial_position", "air_route", "destination"]):
            return data

        # Recursively check each value in the dictionary
        for key in data:
            extracted = flatten_json(data[key])
            if extracted:
                return extracted

    return None


def json_to_xml(json_data):
    root = ET.Element("initial-flightplans", key="initial-flightplans: 69")

    usage = ET.SubElement(root, "usage")
    usage.text = "ALL"
    time = ET.SubElement(root, "time")
    time.text = str(json_data["time"])
    callsign = ET.SubElement(root, "callsign")
    callsign.text = "SQ123"
    rules = ET.SubElement(root, "rules")
    squawk = ET.SubElement(root, "squawk", units="octal")
    squawk.text = "0000"
    aircraft_type = ET.SubElement(root, "type")
    aircraft_type.text = json_data["type"]
    waketurb = ET.SubElement(root, "waketurb")
    waketurb.text = "MEDIUM"
    equip = ET.SubElement(root, "equip")
    vehicle_type = ET.SubElement(root, "vehicle_type")

    dep = ET.SubElement(root, "dep")
    dep_af = ET.SubElement(dep, "af")
    dep_af.text = json_data["departure"]["af"]
    dep_rwy = ET.SubElement(dep, "rwy")

    des = ET.SubElement(root, "des")
    des_af = ET.SubElement(des, "af")
    des_af.text = json_data["destination"]["af"]

    for route in json_data["air_route"]:
        air_route = ET.SubElement(root, "air_route")
        air_route.text = route

    rfl = ET.SubElement(root, "rfl")
    rfl.text = "ALL"
    init = ET.SubElement(root, "init")
    pos = ET.SubElement(init, "pos")
    lat = ET.SubElement(pos, "lat")
    lat.text = json_data["initial_position"]["latitude"]
    lon = ET.SubElement(pos, "lon")
    lon.text = json_data["initial_position"]["longitude"]

    freq = ET.SubElement(init, "freq")
    alt = ET.SubElement(init, "alt", units="")
    alt.text = json_data["initial_position"]["altitude"]
    hdg = ET.SubElement(init, "hdg")
    hdg.text = json_data["initial_position"]["heading"]

    return ET.tostring(root, encoding='unicode')


def main():
    """
    The main function that runs the application.
    """

    page_icon("💬")
    st.subheader("Ollama NARSIM", divider="red", anchor=False)

    client = OpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",  # required, but unused
    )

    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    if available_models:
        selected_model = st.selectbox(
            "Pick a model available locally on your system ↓", available_models
        )

    else:
        st.warning("You have not pulled any model from Ollama yet!", icon="⚠️")
        if st.button("Go to settings to download a model"):
            st.page_switch("pages/03_⚙️_Settings.py")

    message_container = st.container(height=500, border=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        avatar = "🤖" if message["role"] == "assistant" else "😎"
        with message_container.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter a prompt here..."):
        try:
            st.session_state.messages.append(
                {"role": "user", "content": prompt})

            message_container.chat_message("user", avatar="😎").markdown(prompt)

            with message_container.chat_message("assistant", avatar="🤖"):
                with st.spinner("model working..."):
                    stream = client.chat.completions.create(
                        model=selected_model,
                        messages=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        stream=True,
                    )
                # stream response
                response = st.write_stream(stream)
            st.session_state.messages.append(
                {"role": "assistant", "content": response})

            json_d = flatten_json(str2dict(response))
            if json_d:
                # print(f"Valid JSON string: {json_str}")
                st.write([json_to_xml(json_d)])
            else:
                print("The string does not contain valid JSON.")
            # st.write(process_stream_response(response))
        except Exception as e:
            st.error(e, icon="⛔️")


if __name__ == "__main__":
    main()
