import numpy as np
import csv
import onnxruntime as ort
import matplotlib.pyplot as plt
import scipy.signal
from scipy.io import wavfile

# Load the ONNX YAMNet model
session = ort.InferenceSession("yamnet.onnx")
input_name = session.get_inputs()[0].name

def run_yamnet_onnx(waveform):
    """Runs the ONNX model and returns the exact same outputs as TensorFlow YAMNet."""
    inputs = {input_name: waveform}
    outputs = session.run(None, inputs)
    
    # Identify outputs by shape: scores is (_, 521), embeddings is (_, 1024), spectrogram is (_, 64)
    scores, embeddings, spectrogram = None, None, None
    for out in outputs:
        if out.shape[1] == 521:
            scores = out
        elif out.shape[1] == 1024:
            embeddings = out
        else:
            spectrogram = out
            
    return scores, embeddings, spectrogram

def class_names_from_csv(class_map_csv_path):
  """Returns list of class names corresponding to score vector."""
  class_names = []
  with open(class_map_csv_path, 'r', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
      class_names.append(row['display_name'])

  return class_names

class_names = class_names_from_csv("yamnet_class_map.csv")

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    desired_length = int(round(float(len(waveform)) /
                               original_sample_rate * desired_sample_rate))
    waveform = scipy.signal.resample(waveform, desired_length)
  return desired_sample_rate, waveform