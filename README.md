# TTS Voice Cloner

TTS Voice Cloner is a web application that allows users to clone voices and generate text-to-speech audio using advanced AI models.

## Features

- Upload and process reference audio
- Automatic transcription of reference audio
- Text-to-speech generation using F5-TTS or E2-TTS models
- Custom prompt input for generated speech
- Audio playback and download

## Technologies Used

- Backend: Python, Flask
- Frontend: HTML, JavaScript, Tailwind CSS
- AI Models: F5-TTS, E2-TTS
- Audio Processing: librosa, soundfile, pydub
- Transcription: faster-whisper

## Audio Clip Size and Performance

The application supports reference audio clips ranging from 1 second to 25 seconds in length. This range is optimized for the best performance of the F5-TTS and E2-TTS models. While users can use longer audio clips, the results may not be as desirable or consistent.

For optimal results, it's recommended to use reference audio within the 1-25 second range. The application includes functionality to process longer audio files, but users should be aware that exceeding the recommended length might impact the quality of the voice cloning and generated speech.

## Setup and Installation

1. Clone the repository:   ```
   git clone https://github.com/ThisModernDay/f5-tts.git
   cd f5-tts   ```

2. Create and activate a new Conda environment with Python 3.10:   ```
   conda create -n tts-voice-cloner python=3.10
   conda activate tts-voice-cloner   ```

3. Install the required packages:   ```
   pip install -r requirements.txt   ```

4. Set up the environment variables (if necessary).

5. Run the Flask application:   ```
   python app.py   ```

6. Open a web browser and navigate to `http://localhost:5000`.

## Usage

1. Upload a reference audio file (WAV or MP3, ideally between 1-25 seconds).
2. The application will automatically transcribe the audio.
3. Enter your desired prompt text.
4. Choose between F5-TTS and E2-TTS models.
5. Click "Generate Audio" to create the cloned voice audio.
6. Play the generated audio or download it.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
