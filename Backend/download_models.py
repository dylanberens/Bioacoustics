import os
from transformers import ASTConfig, ASTForAudioClassification, ASTFeatureExtractor

def download():
    checkpoint = "MIT/ast-finetuned-audioset-10-10-0.4593"
    print(f"Pre-loading {checkpoint} from Hugging Face . . .")

    ASTFeatureExtractor.from_pretrained(checkpoint)
    ASTForAudioClassification.from_pretrained(checkpoint)
    print("Base weights cached successfully.")

if __name__ == "__main__":
    download()