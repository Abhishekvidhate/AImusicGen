from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import torchaudio
from audiocraft.models import MusicGen
import torch
import os

app = FastAPI()

class GenerateMusicRequest(BaseModel):
    description: str
    duration: int
    model: str

@app.post("/generate_music")
async def generate_music(request: GenerateMusicRequest, file: UploadFile = File(None)):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MusicGen.get_pretrained(request.model).to(device)
        model.set_generation_params(
            use_sampling=True,
            top_k=250,
            duration=request.duration
        )

        if request.model == "facebook/musicgen-melody" and file is not None:
            # Load the uploaded audio file for melody
            melody_wav, melody_sample_rate = torchaudio.load(file.file)
            melody_wav = melody_wav.to(device)
            output = model.generate_with_chroma(
                descriptions=[request.description],
                melody_wavs=[melody_wav],
                melody_sample_rate=melody_sample_rate,
                progress=True,
                return_tokens=True
            )
        else:
            # Generate music without melody
            output = model.generate(
                descriptions=[request.description],
                progress=True,
                return_tokens=True
            )

        samples = output[0].to(device)

        # Save the generated music
        sample_rate = 32000
        save_path = "audio_output/"
        os.makedirs(save_path, exist_ok=True)

        if samples.dim() == 2:
            samples = samples[None, ...]

        audio_path = os.path.join(save_path, f"generated_music.wav")
        torchaudio.save(audio_path, samples[0].cpu(), sample_rate)

        return {"audio_path": audio_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
