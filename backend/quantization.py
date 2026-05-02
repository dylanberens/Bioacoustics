import os
import torch
import torch.nn as nn
from transformers import ASTForAudioClassification

# base custom model definition
class BioAcousticAST(nn.Module):
    def __init__(self, pretrained_model_name):
        super(BioAcousticAST, self).__init__()
        self.ast = ASTForAudioClassification.from_pretrained(
            pretrained_model_name,
            ignore_mismatched_sizes=True,
            attn_implementation="eager"
        )
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
        outputs = self.ast.base_model(input_values, output_attentions=output_attentions)
        last_hidden_state = outputs.last_hidden_state
        cls_token = last_hidden_state[:, 0, :]
        prediction = self.regressor(cls_token)
        if output_attentions:
            return prediction, outputs.attentions
        return prediction, None
    
def quantize_and_save():
    # CRITICAL: dynamic quantization in pytorch must be done on CPU
    device = torch.device("cpu")

    # explicitly set quantization engine to qnnpack for Mac/ARM
    torch.backends.quantized.engine = 'qnnpack'

    pretrained_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    original_path = "checkpoints/best_ast_model_feb.pth"
    quantized_path = "checkpoints/quantized_ast_int8.pth"

    print("1. Loading original float32 model . . . ")
    model = BioAcousticAST(pretrained_name).to(device)
    model.load_state_dict(torch.load(original_path, map_location=device))
    model.eval()

    print("2. Applying Dynamic INT8 Quantization . . .")
    # transformers are heavily bottlenecked by Linear layers
    # we target nn.Linear, casting the weights to int8
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    print("3. Saving quantized model weights . . . ")
    # save the state dict of the newly quantized model
    os.makedirs(os.path.dirname(quantized_path), exist_ok=True)
    torch.save(quantized_model.state_dict(), quantized_path)

    # 4 verify the size reduction
    orig_size = os.path.getsize(original_path) / (1024 * 1024)    
    quant_size = os.path.getsize(quantized_path) / (1024 * 1024)

    print("\n=== Quantization Results ===")
    print(f"Original Size: {orig_size:.2f} MB")
    print(f"Quantized Size: {quant_size:.2f} MB")
    print(f"Reduction: {((orig_size - quant_size) / orig_size) * 100:.1f} %")

if __name__ == "__main__":
    quantize_and_save()