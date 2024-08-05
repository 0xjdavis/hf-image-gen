import streamlit as st

import os
import requests
from transformers import pipeline
import tensorflow
import tensorrt

HUGGINGFACE_KEY = st.secrets['huggingface_key']

def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-large")

    text = image_to_text(url)[0]["generated_text"]

    print("Result:", text)
    return text


def generateStory(scenario):
    template = """
        You are a skilled storyteller. Your task is to create a short, engaging story based on the provided context. 
        The story should be imaginative and concise, not exceeding 30 words.
        Context: {scenario}
        Example:
        Context: They are having a conversation at a table with a cup of coffee.
        Story: "And then she whispered a secret that changed everything," he said over coffee, eyes gleaming.
        Story:
    """

    prompt = template.format(scenario=scenario)
    story_generator = pipeline("text-generation", model="falcon_7b") #gpt2

    story = story_generator(prompt, max_new_tokens=40, num_return_sequences=1)[0]['generated_text']

    print("Story:", story)
    return story

def text2speech(message):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}
    print(inputText)

    if inputText == "error":
        return "An error has occured on Hugging Face."

    # GPT 2
    # gpt2_xl = "https://api-inference.huggingface.co/models/gpt2-xl"
    # gpt2 = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"

    # FALCON 7B
    falcon_7b = "https://api-inference.huggingface.co/models/tiiuae/falcon-7b-instruct"
    falcon_text_for_story = f"create a positive, real, practical and short story from this context {inputText}."

    API_URL = "falcon_7b"
    falcon_text_for_story = f"create a positive, real, practical and short story from this context {inputText}."

    payloads ={
         "inputs": message
    }
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open("audio.flac", "wb") as file:
        file.write(response.content)

#scenario = img2text("image.png")
#story = generateStory(scenario)
#text2speech(story)

def main():
    st.set_page_config(
        page_title="image", page_icon="🤖"
    )
    st.header("Upload an image to create an audio story!")
    uploaded_file=st.file_uploader("Upload an image", type="png")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data=uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        st.image(
            uploaded_file, 
            caption="Uploaded Image", 
            use_column_width= True
            )
        
        scenario = img2text(uploaded_file.name)
        story = generateStory(scenario)
        text2speech = story

        with st.expander("Scenario: "):
            st.write(scenario)

        with st.expander("Story"):
            st.write(story)
        
        st.audio("audio.flac")

main()
