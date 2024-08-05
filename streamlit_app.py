import streamlit as st
import os
import requests
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
import torch

HUGGINGFACE_KEY = st.secrets['huggingface_key']

def img2text(url):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = processor(images=url, return_tensors="pt").pixel_values
    text = model.generate(image)
    generated_text = processor.decode(text[0], skip_special_tokens=True)
    print("Result:", generated_text)
    return generated_text

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
    story_generator = pipeline("text-generation", model="gpt2", framework="pt")
    story = story_generator(prompt, max_new_tokens=40, num_return_sequences=1)[0]['generated_text']
    print("Story:", story)
    return story

def text2speech(message):
    headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}
    API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h-lv60-self"
    payloads = {"inputs": message}
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open("audio.flac", "wb") as file:
        file.write(response.content)

def main():
    st.set_page_config(page_title="Image to Story", page_icon="🤖")
    st.header("Upload an image to create an audio story!")
    uploaded_file = st.file_uploader("Upload an image", type="png")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with open(uploaded_file.name, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        scenario = img2text(uploaded_file.name)
        story = generateStory(scenario)
        text2speech(story)

        with st.expander("Scenario:"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)
        st.audio("audio.flac")

if __name__ == "__main__":
    main()
