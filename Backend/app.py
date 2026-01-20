import os
import io
import base64
import subprocess
import tempfile
import torch
import torch.nn as nn
import numpy as np
import librosa
import librosa.display
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import ASTConfig, ASTForAudioClassification, ASTFeatureExtractor

# initialize flask app
app = Flask(__name__)
CORS(app) # enable CORS for frontend communication

# === 1. CONFIGURATION ===
# docker container wont have GPU usually, defaulting to cpu is safe
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "checkpoints/best_ast_model.pth"
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
CHUNK_DURATION = 10.24
MAX_TOTAL_DURATION = int(SAMPLE_RATE * DURATION)

# === 2. MODEL DEF & VIZ LOGIC ===
class BioAcousticAST(nn.Module):
  def __init__(self, pretrained_model_name):
    super(BioAcousticAST, self).__init__()

    # pretrained body
    self.ast = ASTForAudioClassification.from_pretrained(
        pretrained_model_name,
        ignore_mismatched_sizes=True
    )

    # custom regression head
    hidden_size = self.ast.config.hidden_size
    self.classifier = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, 256),
        nn.GELU(),
        nn.Dropout(0.35),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

  def forward(self, input_values):
    outputs = self.ast.base_model(input_values)
    last_hidden_state = outputs.last_hidden_state
    cls_token = last_hidden_state[:, 0, :]
    prediction = self.classifier(cls_token)
    return prediction

def generate_attention_rollout(model, input_values):
  model.eval()
  with torch.no_grad():
    _, attentions = model(input_values, output_attentions=True)

    seq_len = attentions[0].shape[-1]
    rollout = torch.eye(seq_len).to(DEVICE)
    for layer_attention in attentions:
      avg_head_map = layer_attention.mean(dim=1)[0]
      a_map = avg_head_map + torch.eye(seq_len).to(DEVICE)
      a_map = a_map / a_map.sum(dim=-1, keepdim=True)
      rollout = torch.matmul(rollout, a_map)
    
    cls_attention = rollout[0, 2:]
    grid_h = 12
    grid_w = cls_attention.shape[0] // grid_h
    heatmap = cls_attention[:grid_h*grid_w].reshape(grid_h, grid_w)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return (heatmap.cpu().numpy() * 255).astype(np.uint8)
  
def get_distribution_json(user_score, all_scores):
  # 1. generate heatmap bins
  counts, bin_edges = np.histogram(all_scores, bins=20, range=(0, 1))
  bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

  # 2 hardcoded benchmarks (belong in config file later)
  benchmarks = [
    {"name": "Amazon Rainforest Avg", "value": 0.82},
    {"name": "Urban Park", "value": 0.45},
    {"name": "City Avg", "value": 0.25}
  ]

  return {
    "histogram": {
      "x": bin_centers.tolist(),
      "y": counts.tolist()
    },
    "user_score": {
      "x": [user_score, user_score],
      "y": [0, int(np.max(counts))]
    },
    "benchmarks": [
      {
        "x": [b["value"], b["value"]],
        "y": [0, int(np.max(counts))],
        "name": b["name"]
      } for b in benchmarks
    ]
  }

# ===== 3. PREDICTION PIPELINE =====
def run_full_analysis(file_path, model, feature_extractor):
  audio_full, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_TOTAL_DURATION)
  samples_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION)

  chunk_scores = []
  heatmaps = []
  # sliding window (non-overlapping for speed)
  for start in range(0, len(audio_full) - samples_per_chunk + 1, samples_per_chunk):
    chunk = audio_full[start : start + samples_per_chunk]
    inputs = feature_extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensor="pt")
    input_values = input['input_values'].to(DEVICE)
    
    with torch.no_grad():
      pred, _ = model(input_values)
      score = pred.item()
      raw_heatmap = generate_attention_rollout(model, input_values)

    chunk_scores.append(score)
    heatmaps.append(raw_heatmap * score)

  final_score = np.mean(sorted(chunk_scores, reverse=True)[:3]) if chunk_scores else 0.0

  # stitch heatmaps
  full_heatmap = np.concatenate(heatmaps, axis=1) if heatmaps else np.zeros((12, 100))

  # generate plots (spectrogram & heatmap overlay)
  spec = librosa.feature.melspectrogram(y=audio_full, sr=SAMPLE_RATE, fmax=8000)
  spec_db = librosa.power_to_db(spec, ref=np.max)

  # 1. Base spectrogram
  fig, ax = plt.subplots(figsize=(12, 3))
  librosa.display.specshow(spec_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel', fmax=8000, ax=ax, camp='magma')
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight')
  spec_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
  plt.close()

  # 2. heatmap overlay
  fig, ax = plt.subplots(figsize=(12, 3))
  librosa.display.specshow(spec_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel', fmax=8000, ax=ax, camp='gray')
  heatmap_resized = cv2.resize(full_heatmap, (spec_db.shape[1], spec_db.shape[0]))
  ax.imshow(heatmap_resized, cmap='jet', alpha=0.5, aspect='auto', extent=[0, len(audio_full)/sr, 0, 8000], origin='lower')
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight')
  heat_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
  plt.close()

  return final_score, spec_b64, heat_b64, len(audio_full)/sr

@app.route('/analyze', methods=['POST'])
def analyze():

  if 'audio' not in request.files: return jsonify({"error": "No audio"}), 400
  file = request.files['audio']

  with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    file.save(tmp.name)
    input_path = tmp.name
  
  processed_path = input_path + "_converted.wav"

  try:
    # TRIMMING & CONVERTING W/ FFMEG (solves format & duration issues)
    subprocess.run([
      "ffmpeg", "-y", "-i", input_path,
      "-t", str(MAX_TOTAL_DURATION),
      "-ar", str(SAMPLE_RATE), "-ac", "1",
      processed_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    score, spec_b64, heat_b64, duration = run_full_analysis(processed_path, model, feature_extractor)

    dist_json = get_distribution_json(score, all_normalized_scores)

    return jsonify({
        "biodiversity_score": score,
        "spectrogram_b64": spec_b64,
        "gradcam_b64": heat_b64, # named gradcam just to amtch Frontend
        "distribution_data": dist_json,
        "duration": round(duration, 2),
        "status": "success"
    })
  finally:
    for p in [input_path, processed_path]:
      if os.path.exists(p):
        os.remove(p)

print("Loading Audio Spectrogram Transformer (AST) . . .")
feature_extractor = ASTFeatureExtractor.from_pretrained(PRETRAINED_MODEL)
model = BioAcousticAST(PRETRAINED_MODEL).to(DEVICE)
if os.path.exists(CHECKPOINT_PATH):
  model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)