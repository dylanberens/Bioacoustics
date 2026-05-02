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
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import ASTConfig, ASTForAudioClassification, ASTFeatureExtractor
import urllib.request

# initialize flask app
app = Flask(__name__)
CORS(app) # enable CORS for frontend communication

# === 1. CONFIGURATION ===
CHECKPOINT_PATH = "checkpoints/best_ast_model_feb.pth"
PRETRAINED_MODEL = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
CHUNK_DURATION = 10.24
MAX_TOTAL_DURATION_SECONDS = 60.0

BASELINE_HISTOGRAM_Y = [3, 3, 0, 1, 0, 1, 1, 1, 0, 1, 1, 3, 9, 3, 6, 8, 3, 13, 12, 14, 29, 22, 20, 35, 35, 44, 48, 54, 67, 65, 102, 103, 129, 147, 136, 185, 176, 165, 216, 175, 214, 219, 196, 173, 158, 168, 159, 145, 169, 152]
BIN_CENTERS = np.linspace(0.01, 0.99, 50).tolist()

# === 2. MODEL DEF & VIZ LOGIC ===
class BioAcousticAST(nn.Module):
  def __init__(self, pretrained_model_name):
    super(BioAcousticAST, self).__init__()

    # 1 download only config json, not weights-- eliminate 300MB of downloading that is thrown away
    config = ASTConfig.from_pretrained(pretrained_model_name)
    config.ignore_mismatched_sizes = True
    config._attn_implementation = "eager" # required for attention rollout, hidden attribute- needs leading _ to bypass SDPA default

    # 2. build empty skeleton with random initialization
    self.ast = ASTForAudioClassification(config)

    # custom regression head
    hidden_size = self.ast.config.hidden_size

    self.regressor = nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, 256),
        nn.GELU(),
        nn.Dropout(0.35),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )

  def forward(self, input_values, output_attentions=False):
    # pass thru base transformer
    outputs = self.ast.base_model(input_values, output_attentions=output_attentions)

    # get CLS token (for rollout)
    last_hidden_state = outputs.last_hidden_state
    cls_token = last_hidden_state[:, 0, :]

    # pass thru custom head
    prediction = self.regressor(cls_token)

    if output_attentions:
      # prevents "not enough values to unpack" error
      return prediction, outputs.attentions

    # return dummy None for loss
    return prediction, None

def generate_attention_rollout(model, input_values):
  with torch.no_grad():
    prediction, attentions = model(input_values, output_attentions=True)

    seq_len = attentions[0].shape[-1]
    rollout = torch.eye(seq_len).to(DEVICE)
    for layer_attention in attentions:
      avg_head_map = layer_attention.mean(dim=1)[0]
      a_map = avg_head_map + torch.eye(seq_len).to(DEVICE)
      a_map = a_map / a_map.sum(dim=-1, keepdim=True)
      rollout = torch.matmul(rollout, a_map)
    
    # skip [CLS] & [DIST] tokens to get patch embeddings needed for attention rollout
    cls_attention = rollout[0, 2:]
    grid_h = 12 # (stride of 10, overlap of 6 since patch size = 16, so 12 patches)
    grid_w = cls_attention.shape[0] // grid_h
    heatmap = cls_attention[:grid_h*grid_w].reshape(grid_h, grid_w)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return (heatmap.cpu().numpy() * 255).astype(np.uint8)
  
def get_distribution_json(user_score):
  
  max_y = max(BASELINE_HISTOGRAM_Y)

  return {
    "histogram": {
      "x": BIN_CENTERS,
      "y": BASELINE_HISTOGRAM_Y
    },
    "user_score": {
      "x": [user_score, user_score],
      "y": [0, max_y]
    },
    "benchmarks": [
      {
        "name": "Amazon Rainforest Mean",
        "x": [0.76, 0.76],
        "y": [0, max_y],
        "type": "scatter",
        "mode": "lines",
        "line": {"dash": "dash", "width": 2}
      },
      {
        "name": "Degraded (<1% Amazon)",
        "x": [0.3146, 0.3146],
        "y": [0, max_y],
        "type": "scatter",
        "mode": "lines",
        "line": {"dash": "dash", "width": 2}
      }
    ]
  }

# ===== 3. PREDICTION PIPELINE =====
def run_full_analysis(file_path, model, feature_extractor):
  audio_full, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_TOTAL_DURATION_SECONDS)
  samples_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION)

  chunk_scores = []
  heatmaps = []
  # sliding window (non-overlapping for speed)
  for start in range(0, len(audio_full) - samples_per_chunk + 1, samples_per_chunk):
    chunk = audio_full[start : start + samples_per_chunk]
    inputs = feature_extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs['input_values'].to(DEVICE)
    
    with torch.no_grad():
      prediction, _ = model(input_values)
      score = prediction.item()
      raw_heatmap = generate_attention_rollout(model, input_values)

    chunk_scores.append(score)
    heatmaps.append(raw_heatmap * score)

  final_score = np.mean(sorted(chunk_scores, reverse=True)[:3]) if chunk_scores else 0.0

  # stitch heatmaps
  full_heatmap = np.concatenate(heatmaps, axis=1) if heatmaps else np.zeros((12, 100))

  # generate plots (spectrogram & heatmap overlay)
  spec = librosa.feature.melspectrogram(y=audio_full, sr=SAMPLE_RATE, fmax=8000)
  spec_db = librosa.power_to_db(spec, ref=np.max)

  # 60s file fits 5 complete 10.24s chunks (6th would exceed bounds)
  heatmap_duration = len(heatmaps) * CHUNK_DURATION # 5 * 10.24 = 51.2
  
  # 1. Base spectrogram
  fig, ax = plt.subplots(figsize=(12, 3))
  librosa.display.specshow(spec_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel', fmax=8000, ax=ax, cmap='magma', zorder=1)
  ax.set_xlim(0, heatmap_duration) # only show the 51.2s the model actually analyzes during inference
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight')
  spec_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
  plt.close()

  # 2. heatmap overlay
  fig, ax = plt.subplots(figsize=(12, 3))
  librosa.display.specshow(spec_db, sr=SAMPLE_RATE, x_axis='time', y_axis='mel', fmax=8000, ax=ax, cmap='gray', zorder=1)

  ax.set_xlim(0, heatmap_duration) # only show the 51.2s the model actually analyzes during inference

  # A. calculate exact number of spectrogram frames using actual audio duration
  actual_audio_duration = len(audio_full) / SAMPLE_RATE
  frames_to_keep = int(spec_db.shape[1] * (heatmap_duration / actual_audio_duration))

  # B. resize heatmap to exactly match the target mel spectrogram dimensions (analyzed_frames x 128 mel bins)
  heatmap_resized = cv2.resize(full_heatmap, (frames_to_keep, spec_db.shape[0]))

  # C. plot using librosas specshow, use y_axis mel for exact same log warping to heatmap as was used for background
  librosa.display.specshow(
    heatmap_resized,
    sr=SAMPLE_RATE,
    x_axis='time',
    y_axis='mel',
    fmax=8000,
    ax=ax,
    cmap='jet',
    alpha=0.55,
    zorder=10
  )
  
  buf = io.BytesIO()
  plt.savefig(buf, format='png', bbox_inches='tight')
  heat_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
  plt.close()

  return final_score, spec_b64, heat_b64, len(audio_full)/sr

@app.route('/health', methods=['GET'])
def health():
  return jsonify({"status": "healthy", "model": "AST-Bioacoustics"}), 200

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
      "-t", str(MAX_TOTAL_DURATION_SECONDS),
      "-ar", str(SAMPLE_RATE), "-ac", "1",
      processed_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    score, spec_b64, heat_b64, duration = run_full_analysis(processed_path, model, feature_extractor)

    dist_json = get_distribution_json(score)

    return jsonify({
        "adi_score": score,
        "biodiversity_score": score,
        "spectrogram_b64": spec_b64,
        "gradcam_b64": heat_b64, # named gradcam just to match Frontend
        "distribution_data": dist_json,
        "duration": round(duration, 2),
        "file_size_mb": round(os.path.getsize(processed_path) / (1024 * 1024), 2), # in MB
        "sample_rate": 16000, #hardcoded model requirement
        "status": "success"
    })
  
  except Exception as e:
    print(f"❌ Analysis error: {str(e)}")
    return jsonify({
      "error": "ok i did NOT like that: The file may be corrupted, too short, or in an unsupported format.",
      "status": "error"
    }), 500
  
  finally:
    for p in [input_path, processed_path]:
      if os.path.exists(p):
        os.remove(p)

def download_model(public_url, local_path):
  if not os.path.exists(local_path):
    print(f"Downloading models from public storage . . .")
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    urllib.request.urlretrieve(public_url, local_path)
    print("Model downloaded successfully, huge dub")
  else:
    print("Model already cached locally, skipping download")
  
# ===== QUANTIZATION CONFIG =====
# auto detect engine based on architecture to prevent NoQEngine error locally/on cloud
if torch.backends.quantized.supported_engines == ['qnnpack']:
  torch.backends.quantized.engine = 'qnnpack'
elif 'fbgemm' in torch.backends.quantized.supported_engines:
  torch.backends.quantized.engine = 'fbgemm'

# 1. force cpu for standard quantized inference
DEVICE = torch.device("cpu")

MODEL_URL = "https://storage.googleapis.com/bioacoustics-models/quantized_ast_int8.pth"
CHECKPOINT_PATH = "checkpoints/quantized_ast_int8.pth"

print("Loading Audio Spectrogram Transformer (AST) . . .")
feature_extractor = ASTFeatureExtractor.from_pretrained(PRETRAINED_MODEL)

download_model(MODEL_URL, CHECKPOINT_PATH)

# 2. initialize the base float32 architecture
model = BioAcousticAST(PRETRAINED_MODEL).to(DEVICE)

# 3. mutate architecture to accept INT8 weights
model = torch.quantization.quantize_dynamic(
  model,
  {nn.Linear},
  dtype=torch.qint8
)

# 4. load the quantized weights
if os.path.exists(CHECKPOINT_PATH):
  model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

model.eval()
print("Quantized INT8 model loaded and ready to go")

if __name__ == '__main__':
  #app.run(host='0.0.0.0', port=5000)
  app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))