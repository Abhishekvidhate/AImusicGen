from audiocraft.models import MusicGen
import streamlit as st
import torch
import torchaudio
import os
import numpy as np
import base64


@st.cache_resource
def load_model(model_name):
    model = MusicGen.get_pretrained(model_name)
    return model


def generate_music(description, duration, model_name):
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model(model_name)

    model.set_generation_params(
        use_sampling=True,
        top_k=5,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )

    return output[0]


def generate_music_with_melody(description, melody_wav, melody_sample_rate, duration: int, model_name):
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model(model_name)

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate_with_chroma(
        descriptions=[description],
        melody_wavs=[melody_wav],
        melody_sample_rate=melody_sample_rate,
        progress=True,
        return_tokens=True
    )

    return output[0]


def save_audio(samples: torch.Tensor):
    """Renders an audio player for the given audio samples and saves them to a local directory.

    Args:
        samples (torch.Tensor): a Tensor of decoded audio samples
            with shapes [B, C, T] or [C, T]
    """
    print("Samples (inside function): ", samples)
    sample_rate = 32000
    save_path = "audio_output/"
    os.makedirs(save_path, exist_ok=True)
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


st.set_page_config(
    page_icon="🎵",
    page_title="Music Gen"
)


def main():
    st.title("Text and Melody to Music Generator🎵")

    with st.expander("See explanation"):
        st.write("Music Generator app built using Meta's Audiocraft library. We are using the Music Gen Small model.")

    model_choice = st.selectbox(
        "Select a model",
        ["musicgen-small", "musicgen-melody"],
        index=0
    )

    model_name = f"facebook/{model_choice}"

    text_area = st.text_area("Enter your description.......")
    time_slider = st.slider("Select time duration (In Seconds)", 0, 20, 10)

    if model_choice == "musicgen-melody":
        audio_file = st.file_uploader("Upload a melody audio file (WAV format)", type=["wav"])

    if st.button("Submit"):
        if text_area and time_slider:
            st.json({
                'Your Description': text_area,
                'Selected Time Duration (in Seconds)': time_slider,
                'Selected Model': model_choice
            })

            if model_choice == "musicgen-melody" and audio_file:
                # Load the uploaded audio file
                melody_wav, melody_sample_rate = torchaudio.load(audio_file)
                music_tensors = generate_music_with_melody(text_area, melody_wav, melody_sample_rate, time_slider, model_name)
            elif model_choice == "musicgen-small":
                music_tensors = generate_music(text_area, time_slider, model_name)
            else:
                st.error("Please upload a melody file for the musicgen-melody model.")
                return

            print("Music Tensors: ", music_tensors)
            save_audio(music_tensors)
            audio_filepath = 'audio_output/audio_0.wav'
            audio_file = open(audio_filepath, 'rb')
            audio_bytes = audio_file.read()
            st.audio(audio_bytes)
            st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)
        else:
            st.error("Please enter a description and select a duration. For musicgen-melody, also upload a melody file.")


if __name__ == "__main__":
    main()
