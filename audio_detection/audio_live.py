import pyaudio
import numpy as np
import time
from audio import run_yamnet_onnx, class_names

# YAMNet parameters
SAMPLE_RATE = 16000
CHUNK_SIZE = int(SAMPLE_RATE * 0.975) # Exact 1-second frames needed

def listen_and_predict():
    p = pyaudio.PyAudio()

    try:
        # Open the microphone stream
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=SAMPLE_RATE,
                        input=True,
                        frames_per_buffer=1024)

        print("🎙️   Listening to your microphone... (Press Ctrl+C to stop)")
        print("-" * 50)

        audio_buffer = np.zeros(0, dtype=np.float32)

        while True:
            # Read a chunk from the microphone.
            data = stream.read(2048, exception_on_overflow=False)
            
            # Convert raw bytes to a numpy array of 16-bit integers
            wav_data = np.frombuffer(data, dtype=np.int16)
            
            # Normalize the audio to the range [-1.0, 1.0] expected by YAMNet
            waveform = wav_data / 32768.0
            waveform = waveform.astype(np.float32)

            audio_buffer = np.append(audio_buffer, waveform)

            # Once we have enough audio for YAMNet, process it
            if len(audio_buffer) >= CHUNK_SIZE:
                chunk_to_process = audio_buffer[:CHUNK_SIZE]
                audio_buffer = audio_buffer[CHUNK_SIZE:] # Keep remainder

                # Run the YAMNet ONNX model
                scores, embeddings, spectrogram = run_yamnet_onnx(chunk_to_process)
                
                # Find the most prominent sounds
                class_scores = np.mean(scores, axis=0)
                
                # Get the top 3 detected sounds
                top_indices = np.argsort(class_scores)[::-1][:3]
                
                # Emergency/Distress classes we care about
                EMERGENCY_CLASSES = [
                    "Screaming", "Crying, sobbing", "Wail, moan", "Children shouting",
                    "Alarm", "Alarm clock", "Siren", "Civil defense siren", "Fire alarm",
                    "Smoke detector, smoke alarm", "Gunshot, gunfire", "Explosion", 
                    "Shatter", "Glass", "Breaking"
                ]

                emergency_found = False
                for idx in top_indices:
                    sound = class_names[idx]
                    conf = class_scores[idx]
                    
                    if sound in EMERGENCY_CLASSES and conf > 0.10:
                        print(f"\r🚨 EMERGENCY DETECTED: {sound:<15} | Confidence: {conf:.2f} 🚨{' ' * 10}", end="", flush=True)
                        emergency_found = True
                        break # Only print the highest confidence emergency
                
                if not emergency_found:
                    # Print regular highest-confidence sound
                    top_sound = class_names[top_indices[0]]
                    top_conf = class_scores[top_indices[0]]
                    print(f"\r🔊 Normal: {top_sound:<20} | Confidence: {top_conf:.2f}{' ' * 10}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n⏹️   Stopped listening.")
    
    finally:
        if 'stream' in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()

if __name__ == "__main__":
    listen_and_predict()
