import os
import re
import torch
import torchaudio
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from einops import rearrange
from vocos import Vocos
from pydub import AudioSegment, silence
from model import CFM, UNetT, DiT, MMDiT
from cached_path import cached_path
from model.utils import (
    load_checkpoint,
    get_tokenizer,
    convert_char_to_pinyin,
    save_spectrogram,
)
from faster_whisper import WhisperModel
import librosa
import soundfile as sf
import io
import tempfile
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')
CORS(app)

# --------------------- Settings -------------------- #

target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
target_rms = 0.1
nfe_step = 32  # 16, 32
cfg_strength = 2.0
ode_method = "euler"
sway_sampling_coef = -1.0
speed = 1.0
fix_duration = None

SPLIT_WORDS = [
    "but", "however", "nevertheless", "yet", "still",
    "therefore", "thus", "hence", "consequently",
    "moreover", "furthermore", "additionally",
    "meanwhile", "alternatively", "otherwise",
    "namely", "specifically", "for example", "such as",
    "in fact", "indeed", "notably",
    "in contrast", "on the other hand", "conversely",
    "in conclusion", "to summarize", "finally"
]

# Keep device selection at the top
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# Remove the lazy loading functions and initialize directly
whisper_model = WhisperModel("large", device=device, compute_type="int8")
vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

# Add this near the top of the file, after other imports
UPLOAD_FOLDER = 'data'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_model(repo_name, exp_name, model_cls, model_cfg, ckpt_step):
    ckpt_path = str(cached_path(f"hf://SWivid/{repo_name}/{exp_name}/model_{ckpt_step}.safetensors"))
    vocab_char_map, vocab_size = get_tokenizer("Emilia_ZH_EN", "pinyin")
    model = CFM(
        transformer=model_cls(
            **model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels
        ),
        mel_spec_kwargs=dict(
            target_sample_rate=target_sample_rate,
            n_mel_channels=n_mel_channels,
            hop_length=hop_length,
        ),
        odeint_kwargs=dict(
            method=ode_method,
        ),
        vocab_char_map=vocab_char_map,
    ).to(device)

    model = load_checkpoint(model, ckpt_path, device, use_ema = True)

    return model

# Model configurations
F5TTS_model_cfg = dict(
    dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4
)
E2TTS_model_cfg = dict(dim=1024, depth=24, heads=16, ff_mult=4)

# Dictionary to store loaded models
loaded_models = {}

@app.route('/')
def index():
    logger.info("Accessing index route")
    return render_template('index.html')

@app.route('/upload_audio', methods=['POST'])
def upload_audio():
    logger.info("Accessing upload_audio route")
    if 'audio' not in request.files:
        logger.error("No audio file provided in the request")
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        logger.error("No audio file selected")
        return jsonify({"error": "No audio file selected"}), 400

    if audio_file:
        logger.info(f"Processing audio file: {audio_file.filename}")
        try:
            # Save the file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                audio_file.save(temp_file.name)
                temp_filename = temp_file.name

            logger.info(f"Temporary audio file saved as: {temp_filename}")

            # Load the audio file
            audio, sr = librosa.load(temp_filename, sr=None)

            # Trim silence from the beginning and end
            trimmed_audio, _ = librosa.effects.trim(audio, top_db=20)

            # If the audio is longer than 15 seconds, clip it
            max_duration = 15  # seconds
            if len(trimmed_audio) > max_duration * sr:
                trimmed_audio = trimmed_audio[:max_duration * sr]

            # Save the processed audio
            processed_filename = os.path.join(UPLOAD_FOLDER, audio_file.filename)
            sf.write(processed_filename, trimmed_audio, sr)

            logger.info(f"Processed audio file saved as: {processed_filename}")

            logger.info("Starting transcription with faster-whisper")
            # Transcribe the processed audio file
            segments, info = whisper_model.transcribe(processed_filename)
            logger.info(f"Transcription info: {info}")

            transcription = " ".join([segment.text for segment in segments])
            logger.info(f"Transcription completed. Length: {len(transcription)} characters")
            logger.debug(f"Transcription text: {transcription}")

            # Clean up the temporary file
            os.unlink(temp_filename)

            return jsonify({"message": "Audio uploaded and processed successfully", "transcription": transcription}), 200
        except Exception as e:
            logger.error(f"Audio processing or transcription failed: {str(e)}")
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Audio processing or transcription failed: {str(e)}"}), 500

    return jsonify({"error": "Unknown error occurred"}), 500

@app.route('/generate_audio', methods=['POST'])
def generate_audio():
    logger.info("Accessing generate_audio route")
    data = request.json
    ref_text = data.get('ref_text')
    prompt = data.get('prompt')
    model_choice = data.get('model')
    audio_filename = data.get('audio_filename')  # Add this line to get the filename

    if not all([ref_text, prompt, model_choice, audio_filename]):  # Include audio_filename in the check
        return jsonify({"error": "Missing required parameters"}), 400

    try:
        # Load the model if it's not already loaded
        if model_choice not in loaded_models:
            logger.info(f"Loading {model_choice} model...")
            if model_choice == 'f5-tts':
                loaded_models[model_choice] = load_model(
                    "F5-TTS", "F5TTS_Base", DiT, F5TTS_model_cfg, 1200000
                )
            elif model_choice == 'e2-tts':
                loaded_models[model_choice] = load_model(
                    "E2-TTS", "E2TTS_Base", UNetT, E2TTS_model_cfg, 1200000
                )
            else:
                return jsonify({"error": "Invalid model choice"}), 400

        # Get the loaded model
        model = loaded_models[model_choice]

        # Get the full path of the audio file
        audio_path = os.path.join(UPLOAD_FOLDER, audio_filename)

        if not os.path.exists(audio_path):
            return jsonify({"error": "Audio file not found"}), 404

        # Generate audio
        audio, _ = infer(audio_path, ref_text, prompt, model, False)

        # Save the generated audio to a BytesIO object
        buffer = io.BytesIO()
        sf.write(buffer, audio[1], audio[0], format='wav')
        buffer.seek(0)

        return send_file(buffer, mimetype="audio/wav", as_attachment=True, download_name="generated_audio.wav")

    except Exception as e:
        logger.error(f"Error generating audio: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

def split_text_into_batches(text, max_chars=200, split_words=SPLIT_WORDS):
    if len(text.encode('utf-8')) <= max_chars:
        return [text]
    if text[-1] not in ['。', '.', '!', '！', '?', '？']:
        text += '.'

    sentences = re.split('([。.!?！？])', text)
    sentences = [''.join(i) for i in zip(sentences[0::2], sentences[1::2])]

    batches = []
    current_batch = ""

    def split_by_words(text):
        words = text.split()
        current_word_part = ""
        word_batches = []
        for word in words:
            if len(current_word_part.encode('utf-8')) + len(word.encode('utf-8')) + 1 <= max_chars:
                current_word_part += word + ' '
            else:
                if current_word_part:
                    # Try to find a suitable split word
                    for split_word in split_words:
                        split_index = current_word_part.rfind(' ' + split_word + ' ')
                        if split_index != -1:
                            word_batches.append(current_word_part[:split_index].strip())
                            current_word_part = current_word_part[split_index:].strip() + ' '
                            break
                    else:
                        # If no suitable split word found, just append the current part
                        word_batches.append(current_word_part.strip())
                        current_word_part = ""
                current_word_part += word + ' '
        if current_word_part:
            word_batches.append(current_word_part.strip())
        return word_batches

    for sentence in sentences:
        if len(current_batch.encode('utf-8')) + len(sentence.encode('utf-8')) <= max_chars:
            current_batch += sentence
        else:
            # If adding this sentence would exceed the limit
            if current_batch:
                batches.append(current_batch)
                current_batch = ""

            # If the sentence itself is longer than max_chars, split it
            if len(sentence.encode('utf-8')) > max_chars:
                # First, try to split by colon
                colon_parts = sentence.split(':')
                if len(colon_parts) > 1:
                    for part in colon_parts:
                        if len(part.encode('utf-8')) <= max_chars:
                            batches.append(part)
                        else:
                            # If colon part is still too long, split by comma
                            comma_parts = re.split('[,，]', part)
                            if len(comma_parts) > 1:
                                current_comma_part = ""
                                for comma_part in comma_parts:
                                    if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                        current_comma_part += comma_part + ','
                                    else:
                                        if current_comma_part:
                                            batches.append(current_comma_part.rstrip(','))
                                        current_comma_part = comma_part + ','
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                            else:
                                # If no comma, split by words
                                batches.extend(split_by_words(part))
                else:
                    # If no colon, split by comma
                    comma_parts = re.split('[,，]', sentence)
                    if len(comma_parts) > 1:
                        current_comma_part = ""
                        for comma_part in comma_parts:
                            if len(current_comma_part.encode('utf-8')) + len(comma_part.encode('utf-8')) <= max_chars:
                                current_comma_part += comma_part + ','
                            else:
                                if current_comma_part:
                                    batches.append(current_comma_part.rstrip(','))
                                current_comma_part = comma_part + ','
                        if current_comma_part:
                            batches.append(current_comma_part.rstrip(','))
                    else:
                        # If no comma, split by words
                        batches.extend(split_by_words(sentence))
            else:
                current_batch = sentence

    if current_batch:
        batches.append(current_batch)

    return batches

def infer_batch(ref_audio, ref_text, gen_text_batches, model, remove_silence):
    if ref_audio[0] is not None:
        audio, sr = ref_audio
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        rms = torch.sqrt(torch.mean(torch.square(audio)))
        if rms < target_rms:
            audio = audio * target_rms / rms
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
        audio = audio.to(device)
    else:
        audio = None
        sr = target_sample_rate

    generated_waves = []
    spectrograms = []

    for i, gen_text in enumerate(gen_text_batches):
        # Prepare the text
        if ref_text and len(ref_text[-1].encode('utf-8')) == 1:
            ref_text = ref_text + " "
        text_list = [ref_text + gen_text] if ref_text else [gen_text]
        final_text_list = convert_char_to_pinyin(text_list)

        # Calculate duration
        if audio is not None:
            ref_audio_len = audio.shape[-1] // hop_length
            zh_pause_punc = r"。，、；：？！"
            ref_text_len = len(ref_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, ref_text))
            gen_text_len = len(gen_text.encode('utf-8')) + 3 * len(re.findall(zh_pause_punc, gen_text))
            duration = ref_audio_len + int(ref_audio_len / ref_text_len * gen_text_len / speed)
        else:
            duration = None  # Let the model decide the duration

        # inference
        with torch.inference_mode():
            generated, _ = model.sample(
                cond=audio,
                text=final_text_list,
                duration=duration,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

        if audio is not None:
            generated = generated[:, ref_audio_len:, :]
        generated_mel_spec = rearrange(generated, "1 n d -> 1 d n")
        generated_wave = vocos.decode(generated_mel_spec.cpu())
        if audio is not None and rms < target_rms:
            generated_wave = generated_wave * rms / target_rms

        # wav -> numpy
        generated_wave = generated_wave.squeeze().cpu().numpy()

        generated_waves.append(generated_wave)
        spectrograms.append(generated_mel_spec[0].cpu().numpy())

    # Combine all generated waves
    final_wave = np.concatenate(generated_waves)

    # Remove silence
    if remove_silence:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            sf.write(f.name, final_wave, target_sample_rate)
            aseg = AudioSegment.from_file(f.name)
            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave
            aseg.export(f.name, format="wav")
            final_wave, _ = torchaudio.load(f.name)
        final_wave = final_wave.squeeze().cpu().numpy()

    # Create a combined spectrogram
    combined_spectrogram = np.concatenate(spectrograms, axis=1)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
        spectrogram_path = tmp_spectrogram.name
        save_spectrogram(combined_spectrogram, spectrogram_path)

    return (target_sample_rate, final_wave), spectrogram_path

def infer(ref_audio_orig, ref_text, gen_text, model, remove_silence, custom_split_words=''):
    if not custom_split_words.strip():
        custom_words = [word.strip() for word in custom_split_words.split(',')]
        global SPLIT_WORDS
        SPLIT_WORDS = custom_words

    print(gen_text)

    if ref_audio_orig:
        print("Converting audio...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            aseg = AudioSegment.from_file(ref_audio_orig)

            non_silent_segs = silence.split_on_silence(aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=500)
            non_silent_wave = AudioSegment.silent(duration=0)
            for non_silent_seg in non_silent_segs:
                non_silent_wave += non_silent_seg
            aseg = non_silent_wave

            audio_duration = len(aseg)
            if audio_duration > 15000:
                print("Audio is over 15s, clipping to only first 15s.")
                aseg = aseg[:15000]
            aseg.export(f.name, format="wav")
            ref_audio = f.name
    else:
        ref_audio = None

    if not ref_text.strip():
        if ref_audio:
            print("No reference text provided, transcribing reference audio...")
            segments, info = whisper_model.transcribe(ref_audio, beam_size=5)
            ref_text = " ".join([segment.text for segment in segments])
            print("Finished transcription")
        else:
            print("No reference text or audio provided.")
            ref_text = ""
    else:
        print("Using custom reference text...")

    # Split the input text into batches
    if ref_audio:
        audio, sr = torchaudio.load(ref_audio)
        max_chars = int(len(ref_text.encode('utf-8')) / (audio.shape[-1] / sr) * (30 - audio.shape[-1] / sr))
    else:
        max_chars = 200  # Default value if no reference audio
    gen_text_batches = split_text_into_batches(gen_text, max_chars=max_chars)
    print('ref_text', ref_text)
    for i, gen_text in enumerate(gen_text_batches):
        print(f'gen_text {i}', gen_text)

    print(f"Generating audio using the selected model in {len(gen_text_batches)} batches")
    return infer_batch((audio, sr) if ref_audio else (None, None), ref_text, gen_text_batches, model, remove_silence)

if __name__ == '__main__':
    try:
        print(f"Using {device} device")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        logger.error(traceback.format_exc())