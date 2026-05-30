import os
import io
import base64
import subprocess
import tempfile

# force pytorch to Cloud Run's vCPU limits to prevent thrashing
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import torch
torch.set_num_threads(4) # hard limit pytorch's internal threading
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
  
# === ACTIVATION CACHER CLASS FOR MECHANISTIC INTERPRETABILITY =====

class ActivationCacher:
  def __init__(self, model, target_layers):
    self.model = model
    self.target_layers = target_layers # eg [0, 3, 8] or all 12
    self.activations = {}
    self.hooks = []
    self._register_hooks()

  def _get_activation(self, layer_name):
    def hook(model, input, output):
      # check if output is a tuple (extract the tensor) or already a tensor
      hidden_state = output[0] if isinstance(output, tuple) else output

      # in case it ever drops the batch dimension, force it back
      if hidden_state.ndim == 2:
        hidden_state = hidden_state.unsqueeze(0)

      # output is typically a tuple for HF models; first element is hidden state
      self.activations[layer_name] = hidden_state.detach().cpu()
    return hook

  def _register_hooks(self):
    for i in self.target_layers:
      layer_module = self.model.ast.base_model.encoder.layer[i]
      hook = layer_module.register_forward_hook(self._get_activation(f'layer_{i}'))
      self.hooks.append(hook)

  def remove_hooks(self):
    for hook in self.hooks:
      hook.remove()
  
# ------------------

def isolate_attention_heads(hidden_state, num_heads=12):
  # hidden_state shape: [batch-size, seq_len, 768]
  batch_size, seq_len, hidden_size = hidden_state.shape
  head_dim = hidden_size // num_heads

  # reshape to [batch_size, seq_len, num_heads, head_dim]
  reshaped = hidden_state.view(batch_size, seq_len, num_heads, head_dim)

  # permute to [batch_size, num_heads, seq_len, head_dim] for easier indexing
  return reshaped.permute(0, 2, 1, 3)

# ==========================

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

def enable_mc_dropout(model):
  for m in model.modules():
    if m.__class__.__name__.startswith('Dropout'):
      m.train()

# === MECHANISTIC INTERPRETABILITY EXPANSION ===

def run_mechanistic_inference(audio_path, model, cacher, probes_dict, feature_extractor, top_k=4):
    # runs audio file thru AST and applies probes for mechanistic interpreability

    # 1. prepare audio
    audio, _ = librosa.load(audio_path, sr=16000)

    # pad or truncate to exactly 10.24s expected by AST
    target_len = int(16000 * 10.24)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
    else:
        audio = audio[:target_len]

    # convert to spectrogram features
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs['input_values'].to(DEVICE)

    # 2. forward pass (catch activations)
    cacher.activations.clear()

    with torch.no_grad():
        outputs = model.ast.base_model(input_values, output_attentions=True, return_dict=True)

    all_attentions = outputs.attentions # extract massive tuple of all 12 attention matrices
    results = {}

    # grid parameters based on ast.config
    GRID_H = 12
    GRID_W = 101

    # 3. mechanistic metric & feature space application
    for concept, probe_data in probes_dict.items():
        layer_name = probe_data['layer']
        layer_idx = int(probe_data['layer'].split('_')[1]) # eg 'layer_7' -> 7
        head_idx = probe_data['head']
        
        # load the extracted W and b tensors
        weight = probe_data['weight'].to(DEVICE) # shape [1, 64]
        bias = probe_data['bias'].to(DEVICE) # shape [1]

        # extract target features and apply global max pooling reduction
        raw_acts = cacher.activations[layer_name].to(DEVICE) # [1, 1216, 768]
        heads = isolate_attention_heads(raw_acts, num_heads=12) # [1, 12, 1216, 64]
        pooled_head = heads.max(dim=2)[0] # [1, 12, 64]

        # isolate the 64 dimensional feature vector of this head
        x = pooled_head[0, head_idx, :] # [64]

        # core math: y = sigmoid(W*x + b)
        # unsqueeze x to do matrix multiplication: [1, 64] @ [64, 1] -> [1, 1] linear algebra dog
        logit = torch.matmul(weight, x.unsqueeze(1)).squeeze() + bias
        probability = torch.sigmoid(logit).item()

        # XAI heatmap extraction. shape of single layer attention [Batch, Num_Heads, Seq_Len, Seq_Len]
        layer_attn = all_attentions[layer_idx]

        # isolate specific head, grab [CLS] token's row (index 0). slice [2:] to ignore [CLS] and [DIST] special tokens
        cls_attention = layer_attn[0, head_idx, 0, 2:]

        # min max normalization over active frame arrays for frontend rendering
        attn_min = cls_attention.min()
        attn_max = cls_attention.max()
        if attn_max > attn_min:
          normalized_attn = (cls_attention - attn_min) / (attn_max - attn_min + 1e-8)
        else:
          normalized_attn = torch.zeros_like(cls_attention)

        # convert vector map directly to 2D numpy coordinate plane
        attn_np = normalized_attn.cpu().numpy()

        if attn_np.shape[0] == (GRID_H * GRID_W):
          heatmap_2d = attn_np.reshape(GRID_H, GRID_W)
        else:
          # EH runtime fallback if input dimensions inconsistent
          fallback_w = attn_np.shape[0] // GRID_H
          heatmap_2d = attn_np[:GRID_H * fallback_w].reshape(GRID_H, fallback_w)

        results[concept] = {
            'probability': probability,
            'heatmap_vector': heatmap_2d.tolist() # format directly to serializable list for API delivery
            }

    # 4. sort and return the top K findings
    sorted_results = sorted(results.items(), key=lambda item: item[1]['probability'], reverse=True)
    return sorted_results[:top_k]

# ===== 3. PREDICTION PIPELINE =====
def run_full_analysis(file_path, model, feature_extractor):
  audio_full, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=MAX_TOTAL_DURATION_SECONDS)
  samples_per_chunk = int(SAMPLE_RATE * CHUNK_DURATION)

  chunk_scores = []
  chunk_stds = [] # track variance for interval
  heatmaps = []

  # sliding window (non-overlapping for speed)
  for start in range(0, len(audio_full) - samples_per_chunk + 1, samples_per_chunk):
    chunk = audio_full[start : start + samples_per_chunk]
    inputs = feature_extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_values = inputs['input_values'].to(DEVICE)
    
    with torch.no_grad():
      # 1. turn on MC dropout
      enable_mc_dropout(model)

      # 2. run forward pass 5 times
      mc_predictions = [model(input_values)[0].item() for _ in range(5)]

      # 3. calculate mean and standard deviation
      score = np.mean(mc_predictions)
      uncertainty = np.std(mc_predictions)

      # 4. turn model back to standard eval mode, get the heatmap once
      model.eval()
      raw_heatmap = generate_attention_rollout(model, input_values)

    chunk_scores.append(score)
    chunk_stds.append(uncertainty)
    heatmaps.append(raw_heatmap * score)
  
  # calculate final score (mean of top 3)
  top_3_indices = np.argsort(chunk_scores)[-3:][::-1]
  final_score = np.mean([chunk_scores[i] for i in top_3_indices]) if chunk_scores else 0.0

  # calculate final uncertainty (mean of standard deviations from those top 3 chunks)
  final_std = np.mean([chunk_stds[i] for i in top_3_indices]) if chunk_stds else 0.0

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

  return final_score, final_std, spec_b64, heat_b64, len(audio_full)/sr

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

    score, std_dev, spec_b64, heat_b64, duration = run_full_analysis(processed_path, model, feature_extractor)

    # run mechanistic interpretability
    top_concepts = run_mechanistic_inference(processed_path, model, cacher, probes_dict, feature_extractor, top_k=4)

    # calculate 95% Confidence Interval
    margin_of_error = 2 * std_dev
    lower_bound = max(0.0, score - margin_of_error) # clamp to 0
    upper_bound = min(1.0, score + margin_of_error) # clamp to 1

    dist_json = get_distribution_json(score)

    return jsonify({
        "adi_score": score,
        "confidence_interval": [round(lower_bound, 3), round(upper_bound, 3)],
        "biodiversity_score": score,
        "spectrogram_b64": spec_b64,
        "gradcam_b64": heat_b64, # named gradcam just to match Frontend
        "distribution_data": dist_json,
        "mechanistic_concepts": top_concepts,
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

PROBES_URL = "https://storage.googleapis.com/bioacoustics-models/mechanistic_probes.pt"
PROBES_PATH = "checkpoints/mechanistic_probes.pt"

download_model(PROBES_URL, PROBES_PATH)

# load to CPU
probes_dict = torch.load(PROBES_PATH, map_location=DEVICE)

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

# initialize global cacher for all 12 layers; attach hooks as final step, after model has been built, quantized, loaded weights
cacher = ActivationCacher(model, list(range(12)))

print("Quantized INT8 model loaded and ready to go")

if __name__ == '__main__':
  #app.run(host='0.0.0.0', port=5000)
  app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))