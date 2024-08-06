import streamlit as st
from PIL import Image
import requests
import base64
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
import torch as torch

HUGGINGFACE_KEY = st.secrets['huggingface_key']

def img2text(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    # print("Result", generated_text)
    return scenario

def generateStory(scenario):
    template = "Create a short, engaging story based on the provided scenario. The story should be imaginative and concise, not exceeding 30 words."
    prompt = template.format(scenario=scenario)
    story_generator = pipeline("text-generation", model="gpt2", framework="pt")
    story = story_generator(prompt, max_new_tokens=400, num_return_sequences=1)[0]['scenario']
    # print("Story", story)
    return story

    
def text2speech(message):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    
    inputs = processor(text="My dog is so cute.", return_tensors="pt")
    speaker_embeddings = torch.zeros((1, 512))  # or load xvectors from a file
    
    set_seed(555)  # make deterministic
    
    # generate speech
    speech = model.generate(inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=vocoder)
    speech.shape
    torch.Size([15872])
    
    #headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}
    # API_URL = "https://api-inference.huggingface.co/models/facebook/wav2vec2-large-960h-lv60-self"
    #payload = {"inputs": message}
    #response = requests.post(API_URL, headers=headers, json=payload)
    #with open("story.mpeg", "wb") as file:
    #    file.write(response.content)

def main():
    st.set_page_config(page_title="Image to Story", page_icon="🤖")
    st.header("Upload an image to create an audio story!")
    uploaded_file = st.file_uploader("Upload an image", type="png")

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_path = f"{uploaded_file.name}"
        with open(image_path, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        scenario = img2text(image_path)
        story = generateStory(scenario)
        text2speech(story)

        
        with st.expander("Template"):
            st.write(template)
        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)
        audio_bytes = StringIO(story.decode("utf-8"))
        st.audio(audio_bytes, format='audio/mp3')

if __name__ == "__main__":
    main()
