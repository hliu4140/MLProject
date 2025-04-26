import os
import random
import time
import json
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

from transformers import (
    DistilBertModel, DistilBertPreTrainedModel, DistilBertConfig, DistilBertTokenizer,
    BertModel, BertPreTrainedModel, BertConfig, BertTokenizer,
    get_linear_schedule_with_warmup,
)
from tqdm.notebook import trange, tqdm

class DataPreprocessor:
    def __init__(
        self,
        csv_file: str,
        text_col: str,
        label_cols: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.csv_file     = csv_file
        self.text_col     = text_col
        self.label_cols   = label_cols
        self.test_size    = test_size
        self.random_state = random_state

    def load_and_split(self):
        df = pd.read_csv(self.csv_file)
        train_df, val_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True
        )
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
    
class JRSBaseDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        text_col: str,
        label_cols: List[str],
        max_len: int
    ):
        self.df         = df
        self.tokenizer  = tokenizer
        self.text_col   = text_col
        self.label_cols = label_cols
        self.max_len    = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        toks = self.tokenizer(
            row[self.text_col],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        labels = row[self.label_cols].to_numpy(dtype=float)
        return (
            toks["input_ids"].squeeze(0),
            toks["attention_mask"].squeeze(0),
            toks.get("token_type_ids", torch.zeros_like(toks["input_ids"])).squeeze(0),
            torch.tensor(labels, dtype=torch.float)
        )
    
class AverageMeter:
    """Track average of any metric (e.g. loss)."""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = self.sum = self.count = self.avg = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_lbls = [], []
    with torch.no_grad():
        for input_ids, mask, ttype, labels in loader:
            input_ids, mask, ttype = (
                input_ids.to(device),
                mask.to(device),
                ttype.to(device),
            )
            logits = model(input_ids, attention_mask=mask, token_type_ids=ttype)
            preds = (torch.sigmoid(logits).cpu().numpy() > 0.5).astype(int)
            all_preds.append(preds)
            all_lbls.append(labels.numpy())
    return accuracy_score(
        np.vstack(all_lbls),
        np.vstack(all_preds)
    )

class JRSDistilBERTModel(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.distilbert = DistilBertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0]   # [CLS] token
        return self.classifier(pooled)

class JRSBERTModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert       = BertModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = out.pooler_output
        return self.classifier(pooled)

class JRSHateBERTModel(JRSBERTModel):
    """Same as BERT but loads HateBERT weights."""
    @classmethod
    def from_pretrained(cls, model_name_or_path, *model_args, config=None, **kwargs):
        # if Colab/your trainer passed a config, use it; otherwise load a fresh one
        if config is None:
            config = BertConfig.from_pretrained(model_name_or_path, **kwargs)
        # config.num_labels is already set correctly by your trainer
        return super().from_pretrained(model_name_or_path, config=config, **kwargs)

class CheckpointManager:
    """
    Saves model.state_dict() and metrics (accuracy, loss, cumulative time) each epoch,
    and can reload model weights when needed.
    """
    def __init__(self, save_dir: str):
        self.save_dir     = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history_path = os.path.join(save_dir, "history.json")

        if os.path.exists(self.history_path):
            with open(self.history_path, "r") as f:
                self.history = json.load(f)
        else:
            # {"1": {"accuracy":…, "loss":…, "time":…}, …}
            self.history = {}

    def save(
        self,
        epoch: int,
        model: nn.Module,
        accuracy: float,
        loss: float,
        cum_time: float
    ):
        # record all three metrics
        self.history[str(epoch)] = {
            "accuracy": accuracy,
            "loss":     loss,
            "time":     cum_time
        }

        # save model weights
        ckpt_path = os.path.join(self.save_dir, f"model_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

        # persist JSON
        with open(self.history_path, "w") as f:
            json.dump(self.history, f, indent=2)

    def load_model(
        self,
        model: nn.Module,
        epoch: int,
        map_location=None
    ) -> nn.Module:
        ckpt_path = os.path.join(self.save_dir, f"model_epoch{epoch}.pt")
        state_dict = torch.load(ckpt_path, map_location=map_location)
        model.load_state_dict(state_dict)
        return model

    def get_history(self) -> Dict[int, Dict[str, float]]:
        """Return { epoch: {"accuracy":…, "loss":…, "time":…}, … }"""
        return { int(k): v for k, v in self.history.items() }
    
    def get_best_epoch(self) -> int:
        """
        Return the epoch number that achieved the highest accuracy.
        If no history, raises a ValueError.
        """
        history = self.get_history()  # {epoch: {"accuracy":…, …}}
        if not history:
            raise ValueError("No checkpoints in history.")
        # pick the epoch whose accuracy is maximal
        best_epoch = max(history.items(), key=lambda kv: kv[1]["accuracy"])[0]
        return best_epoch

    def load_best_model(
        self,
        model: nn.Module,
        map_location=None
    ) -> Tuple[nn.Module, int]:
        """
        Load the weights from the epoch with highest accuracy into `model`.
        Returns (model, best_epoch).
        """
        best_epoch = self.get_best_epoch()
        loaded = self.load_model(model, best_epoch, map_location=map_location)
        return loaded, best_epoch

class ModelTrainer:
    """
    Encapsulates fine‑tuning for one Transformer model:
      - loads model + tokenizer
      - runs training loop, eval each epoch
      - checkpoints weights & accuracy via CheckpointManager
      - returns history dict (epoch→accuracy)
    """
    def __init__(
        self,
        model_name: str,
        ModelClass,
        ConfigClass,
        TokenizerClass,
        label_cols: List[str],
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        save_dir: str,
        lr: float = 3e-5,
        epochs: int = 3
    ):
        self.device      = device
        self.epochs      = epochs
        self.train_loader= train_loader
        self.val_loader  = val_loader
        self.save_dir    = save_dir

        # ----- setup model -----
        config = ConfigClass.from_pretrained(model_name)
        config.num_labels = len(label_cols)
        self.model = ModelClass.from_pretrained(model_name, config=config)
        self.model.to(self.device)

        # ----- tokenizer (if you need to do inference later) -----
        self.tokenizer = TokenizerClass.from_pretrained(model_name)

        # ----- optimizer & scheduler -----
        total_steps = len(self.train_loader) * self.epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # ----- checkpoint manager -----
        self.ckpt = CheckpointManager(save_dir)

    def train(self, resume: bool = False) -> Dict[int, Dict[str, float]]:
            # --- handle resume ---
            if resume:
                history = self.ckpt.get_history()
                if history:
                    last_epoch = max(history.keys())
                    print(f"⚡ Resuming from epoch {last_epoch}")
                    self.model = self.ckpt.load_model(self.model, last_epoch)
                    start_epoch = last_epoch + 1
                    cum_time = history[last_epoch]["time"]
                else:
                    start_epoch = 1
                    cum_time     = 0.0
            else:
                start_epoch = 1
                cum_time     = 0.0

            # --- epoch loop ---
            for epoch in trange(start_epoch, self.epochs + 1,
                                desc="Training Epochs", unit="epoch"):

                start_time = time.time()
                meter      = AverageMeter()
                self.model.train()

                for input_ids, mask, ttype, labels in tqdm(
                    self.train_loader,
                    desc=f"Epoch {epoch} batches",
                    leave=False,
                    unit="batch"
                ):
                    input_ids, mask, ttype, labels = (
                        input_ids.to(self.device),
                        mask.to(self.device),
                        ttype.to(self.device),
                        labels.to(self.device),
                    )
                    self.optimizer.zero_grad()
                    logits = self.model(
                        input_ids,
                        attention_mask=mask,
                        token_type_ids=ttype
                    )
                    loss = nn.BCEWithLogitsLoss()(logits, labels)
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    meter.update(loss.item(), input_ids.size(0))

                # compute metrics
                train_loss = meter.avg
                val_acc    = evaluate(self.model, self.val_loader, self.device)
                elapsed    = time.time() - start_time
                cum_time  += elapsed

                print(
                    f"Epoch {epoch}/{self.epochs}  "
                    f"train_loss {train_loss:.4f}  "
                    f"val_acc    {val_acc:.4f}  "
                    f"epoch_time {elapsed:.1f}s  "
                    f"cum_time   {cum_time:.1f}s"
                )

                # checkpoint accuracy, loss, and cumulative time
                self.ckpt.save(epoch, self.model, val_acc, train_loss, cum_time)

            return self.ckpt.get_history()

class MultiHistoryPlotter:
    '''
    Plots accuracy, loss, and cumulative time per epoch
    for multiple models on the same axes.
    '''
    def __init__(self, histories: dict):
        """
        histories: dict mapping model_name -> path_to_history_json
        """
        self.data = {}
        for name, path in histories.items():
            with open(path, 'r') as f:
                raw = json.load(f)
            # convert keys to int
            self.data[name] = {int(k): v for k, v in raw.items()}

    def plot_accuracy(self):
        plt.figure()
        for name, hist in self.data.items():
            epochs = sorted(hist.keys())
            accuracies = [hist[e]['accuracy'] for e in epochs]
            plt.plot(epochs, accuracies, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Validation Accuracy')
        plt.title('Validation Accuracy per Epoch')
        plt.legend()
        plt.show()

    def plot_loss(self):
        plt.figure()
        for name, hist in self.data.items():
            epochs = sorted(hist.keys())
            losses = [hist[e]['loss'] for e in epochs]
            plt.plot(epochs, losses, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Training Loss per Epoch')
        plt.legend()
        plt.show()

    def plot_time(self):
        plt.figure()
        for name, hist in self.data.items():
            epochs = sorted(hist.keys())
            times = [hist[e]['time'] for e in epochs]
            plt.plot(epochs, times, label=name)
        plt.xlabel('Epoch')
        plt.ylabel('Cumulative Time (s)')
        plt.title('Cumulative Training Time per Epoch')
        plt.legend()
        plt.show()

    def plot_all(self):
        self.plot_accuracy()
        self.plot_loss()
        self.plot_time()

# Example usage:
# histories = {
#     'ModelA': '/mnt/data/checkpoints/ModelA/history.json',
#     'ModelB': '/mnt/data/checkpoints/ModelB/history.json',
#     'ModelC': '/mnt/data/checkpoints/ModelC/history.json',
# }
# plotter = MultiHistoryPlotter(histories)
# plotter.plot_accuracy()  # plot accuracy curves for all three
# plotter.plot_all()       # plot all three metrics in sequence

if __name__ == '__main__':
    main()