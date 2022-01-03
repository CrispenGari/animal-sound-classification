from app import app
from flask import make_response, jsonify, request
import io
import soundfile as sf
import torch
import numpy as np
from models import model, predict_label

class AppConfig:
    PORT = 3001
    DEBUG = False
    
    
@app.route('/classify', methods=["POST"])    
def classify_audio():
    data = {"success": False}
    if request.method == "POST":
        if request.files.get("audio"):
            # read the audio in PIL format
            audio = request.files.get("audio").read()
            audio = io.BytesIO(audio)
            waveform, samplerate = sf.read(file=audio, dtype='float32')
            waveform = torch.from_numpy(np.array([waveform]))
            preds = predict_label(model=model, waveform=waveform)
            data["success"] = True
            data["predictions"] = preds    
    return make_response(jsonify(data)), 200
    
    
@app.route('/', methods=["GET"])
def meta():
    meta ={
        "programmer": "@crispengari",
        "main": "audio classification",
        "description": "given an a audio of an animal the model should classify weather the sound is for a cat or a dog.",
        "language": "python",
        "library": "pytorch",
        "mainLibray": "torchaudio"
    }
    return make_response(jsonify(meta)), 200


if __name__ == "__main__":
    app.run(debug=AppConfig().DEBUG, port=AppConfig().PORT, )