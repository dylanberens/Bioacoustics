# used on drive

# import torch
# import librosa
# import numpy as np

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
  
def isolate_attention_heads(hidden_state, num_heads=12):
  # hidden_state shape: [batch-size, seq_len, 768]
  batch_size, seq_len, hidden_size = hidden_state.shape
  head_dim = hidden_size // num_heads

  # reshape to [batch_size, seq_len, num_heads, head_dim]
  reshaped = hidden_state.view(batch_size, seq_len, num_heads, head_dim)

  # permute to [batch_size, num_heads, seq_len, head_dim] for easier indexing
  return reshaped.permute(0, 2, 1, 3)

def run_mechanistic_inference(audio_path, model, cacher, probes_dict, feature_extractor, top_k=4):
    # runs audio file thru AST and applies probes for mechanistic interpreability

    # 1. prepare audio
    audio, _ = librosa.load(audio_path, sr=16000)

    # pad or truncate to exactly 10.24s expected by AST
    target_len = int(16000 * 10.24)
    if len(audio) < target_len:
        audio = np.pad(audio, (0, target_len - len(audio)))
    else:
        audio = audio[:target_len]
    
    # convert to spectrogram features
    inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
    input_values = inputs['input_values'].to('cuda')

    # 2. forward pass (catch activations)
    cacher.activations.clear()
    with torch.no_grad():
        _ = model(input_values)
    
    results = {}

    # 3. apply the mathematical probes
    for concept, probe_data in probes_dict.items():
        layer_name = probe_data['layer']
        head_idx = probe_data['head']

        # load the extracted W and b tensors
        weight = probe_data['weight'].to('cuda') # shape [1, 64]
        bias = probe_data['bias'].to('cuda') # shape [1]

        # laod the pooled activation for this specific head
        raw_acts = cacher.activations[layer_name].to('cuda') # [1, 1216, 768]
        heads = isolate_attention_heads(raw_acts, num_heads=12) # [1, 12, 1216, 64]
        pooled_head = heads.max(dim=2)[0] # [1, 12, 64]

        # isolate the 64 dimensional feature vector of this head
        x = pooled_head[0, head_idx, :] # [64]

        # core math: y = sigmoid(W*x + b)
        # unsqueeze x to do matrix multiplication: [1, 64] @ [64, 1] -> [1, 1] linear algebra dog
        logit = torch.matmul(weight, x.unsqueeze(1)).squeeze() + bias
        probability = torch.sigmoid(logit).item()

        results[concept] = probability
    
    # 4. sort and return the top K findings
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    return sorted_results[:top_k]

probes_dict = torch.load('/content/drive/MyDrive/Bioacoustics/mechanistic_probes.pt', map_location='cuda')

top_predictions = run_mechanistic_inference('/content/drive/MyDrive/rfcx-species-audio-detection/test/f275ae2a1.flac', model, cacher, probes_dict, feature_extractor)

print("Top Discoveries")
for concept, prob in top_predictions:
  print(f"{concept}: {prob*100:1f}%")