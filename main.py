import os
import torch
import librosa
import ffmpeg
from tqdm import tqdm
from pydub import AudioSegment
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# --- PARAMETERS ---
MODEL_NAME = "openai/whisper-medium"
CHUNK_LENGTH_MS = 30000  # 30 seconds
SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".m4v"]
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mkv", ".avi", ".mov", ".m4v"]

# --- DEVICE SELECTION ---
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

# --- EXTRACT AUDIO FROM VIDEO ---
def extract_audio_from_video(video_path, audio_output="temp_audio.wav"):
    try:
        ffmpeg.input(video_path).output(
            audio_output,
            format="wav",
            acodec="pcm_s16le",
            ac=1,
            ar="16k"
        ).run(overwrite_output=True)
        return audio_output
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None

# --- SPLIT AUDIO INTO CHUNKS ---
def split_audio(file_path, chunk_length_ms=CHUNK_LENGTH_MS):
    audio = AudioSegment.from_file(file_path)
    return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

# --- TRANSCRIBE SINGLE CHUNK ---
def transcribe_audio_chunk(chunk, processor, model, device):
    try:
        temp_file = "temp_chunk.wav"
        chunk.export(temp_file, format="wav")
        audio, sr = librosa.load(temp_file, sr=16000)
        audio_tensor = torch.tensor(audio, dtype=torch.float32).to(device)
        inputs = processor(audio_tensor.cpu(), sampling_rate=sr, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        forced_decoder_ids = processor.get_decoder_prompt_ids(language="ru", task="transcribe")
        with torch.no_grad():
            predicted_ids = model.generate(
                input_features,
                max_new_tokens=300,
                num_beams=5,
                do_sample=False,
                forced_decoder_ids=forced_decoder_ids
            )
        os.remove(temp_file)
        return processor.batch_decode(predicted_ids.to("cpu"), skip_special_tokens=True)[0]
    except Exception as e:
        print(f"Error transcribing chunk: {e}")
        return None

# --- BUILD OUTPUT FILENAME ---
def get_output_filename(output_dir, base_name):
    return os.path.join(output_dir, f"{base_name}.txt")

# --- TRANSCRIBE FULL AUDIO FILE ---
def transcribe_audio_whisper(file_path, processor, model, device, output_dir="transcripts", base_name=None):
    try:
        chunks = split_audio(file_path)
        transcription = []
        for chunk in tqdm(chunks, desc="Transcribing"):
            text = transcribe_audio_chunk(chunk, processor, model, device)
            if text:
                transcription.append(text)
        full_transcription = "\n".join(transcription)
        if base_name is None:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
        os.makedirs(output_dir, exist_ok=True)
        output_file = get_output_filename(output_dir, base_name)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(full_transcription)
        return full_transcription
    except Exception as e:
        print(f"Error transcribing {file_path}: {e}")
        return None

# --- FILE TYPE CHECK ---
def is_video_file(file_path):
    return any(file_path.lower().endswith(ext) for ext in SUPPORTED_VIDEO_FORMATS)

def is_audio_file(file_path):
    return any(file_path.lower().endswith(ext) for ext in SUPPORTED_AUDIO_FORMATS)

# --- TRANSCRIBE ALL FILES IN DIRECTORY ---
def transcribe_all_files_in_folder(input_dir="videos", output_dir="transcripts"):
    os.makedirs(output_dir, exist_ok=True)
    entries = os.listdir(input_dir)
    files = [f for f in entries if is_audio_file(f) or is_video_file(f)]
    if not files:
        return
    device = get_device()
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)
    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32 if device == "mps" else torch.float16
    ).to(device)
    for file in files:
        file_path = os.path.join(input_dir, file)
        base_name = os.path.splitext(file)[0]
        if is_video_file(file_path):
            audio_path = extract_audio_from_video(file_path)
            if not audio_path:
                continue
            transcribe_audio_whisper(audio_path, processor, model, device, output_dir, base_name)
        else:
            transcribe_audio_whisper(file_path, processor, model, device, output_dir, base_name)

# --- ENTRY POINT ---
if __name__ == "__main__":
    transcribe_all_files_in_folder("videos", "transcripts")
