import streamlit as st
from PIL import Image
import requests
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, set_seed
import torch
import soundfile as sf  # Add this import for saving audio

HUGGINGFACE_KEY = st.secrets['huggingface_key']

def img2text(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    image = Image.open(image_path)
    inputs = processor(images=image, return_tensors="pt")
    output = model.generate(**inputs)
    generated_text = processor.decode(output[0], skip_special_tokens=True)
    return generated_text
    
# Create a short, engaging story based on the provided scenario. The story should be imaginative and not exceed 30 words. Scenario: 
def generateStory(scenario): 
    template = """
        {scenario}
    """
    prompt = template.format(scenario=scenario)
    story_generator = pipeline("text-generation", model="gpt2-large", framework="pt")
    story = story_generator(prompt, max_new_tokens=40, num_return_sequences=1)[0]['generated_text']
    return story

def text2speech(message):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text=message, return_tensors="pt")
    speaker_embeddings = torch.zeros((1, 512))  # Use default or specific speaker embeddings

    # Make deterministic
    set_seed(555)  

    # Generate speech
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings=speaker_embeddings, vocoder=vocoder)

    # Convert tensor to numpy array and save as a WAV file
    audio_data = speech.detach().cpu().numpy()
    audio_path = "audio.wav"
    sf.write(audio_path, audio_data, samplerate=16000)

    return audio_path

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
        audio_file = text2speech(story)

        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)

        st.audio(audio_file, format='audio/wav')

if __name__ == "__main__":
    main()
