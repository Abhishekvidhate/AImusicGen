# AImusicGen

Welcome to AImusicGen, a project utilizing Meta's [AudioCraft](https://ai.meta.com/resources/models-and-libraries/audiocraft/) for music generation. This repository contains the code and models used for training and generating music.

![AudioCraft](https://ai.meta.com/resources/models-and-libraries/audiocraft/images/audiocraft.png)


## Introduction

AImusicGen explores AI's capabilities in music generation using Meta's AudioCraft. The generated music can serve various purposes, including background scores and experimental compositions.

## Features

- Music generation using Meta's AudioCraft
- Pre-trained models for quick use
- Support for different genres and styles

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Abhishekvidhate/AImusicGen.git
    cd AImusicGen
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To generate music using a pre-trained AudioCraft model, run:

```bash
streamlit run streamlit_app.py
