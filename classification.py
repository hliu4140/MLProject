from train import*
import pandas as pd
import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertConfig, DistilBertTokenizer
from train import CheckpointManager, JRSDistilBERTModel

class TextDataset(Dataset):
    """
    Simple Dataset that tokenizes a list of texts for DistilBERT.
    """
    def __init__(self, texts, tokenizer, max_len):
        self.texts     = texts
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        toks = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        # DistilBERT has no token_type_ids by default, so we zero‑fill if missing
        token_type_ids = toks.get("token_type_ids", torch.zeros_like(toks["input_ids"]))
        return (
            toks["input_ids"].squeeze(0),
            toks["attention_mask"].squeeze(0),
            token_type_ids.squeeze(0)
        )

def predict_and_save_logits(
    csv_path: str,
    save_dir: str,
    output_json: str,
    model_name: str = "distilbert-base-uncased",
    max_len: int = 384,
    batch_size: int = 96,
    num_labels: int = 6
):
    # 1) Read data
    df             = pd.read_csv(csv_path)
    more_texts     = df["more_toxic"].tolist()
    less_texts     = df["less_toxic"].tolist()

    # 2) Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 3) Tokenizer + Model + Best checkpoint
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    config    = DistilBertConfig.from_pretrained(model_name)
    config.num_labels = num_labels

    model = JRSDistilBERTModel(config).to(device)
    ckpt  = CheckpointManager(save_dir)
    model, best_epoch = ckpt.load_best_model(model, map_location=device)
    model.eval()

    # 4) Inference helper
    def get_probs(texts):
        ds = TextDataset(texts, tokenizer, max_len)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        all_probs = []
        with torch.no_grad():
            for input_ids, mask, ttype in dl:
                input_ids, mask, ttype = (
                    input_ids.to(device),
                    mask.to(device),
                    ttype.to(device),
                )
                logits = model(
                    input_ids,
                    attention_mask=mask,
                    token_type_ids=ttype
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    # 5) Run on both columns
    more_probs = get_probs(more_texts)
    less_probs = get_probs(less_texts)

    # 6) Save to JSON
    out = {
        "more_toxic": more_probs.tolist(),
        "less_toxic": less_probs.tolist(),
        "best_epoch": best_epoch
    }
    with open(output_json, "w") as f:
        json.dump(out, f, indent=2)

    print(f"✅ Saved logits (best_epoch={best_epoch}) to {output_json}")

predict_and_save_logits(
    csv_path    = r"Data\JigsawRate\validation_data.csv",
    save_dir    = "checkpoints/DistilBERT",
    output_json = "distilbert_logits.json",
    model_name  = "distilbert-base-uncased",
    max_len     = 384,
    batch_size  = 96,
    num_labels  = 6
)

def main():
    df = pd.read_csv(r"Data\JigsawRate\validation_data.csv")
    predict_and_save_logits(
        csv_path    = r"Data\JigsawRate\validation_data.csv",
        save_dir    = "checkpoints/DistilBERT",
        output_json = "distilbert_logits.json",
        model_name  = "distilbert-base-uncased",
        max_len     = 384,
        batch_size  = 96,
        num_labels  = 6
    )
if __name__ == '__main__':
    main()