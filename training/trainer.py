import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import os
import time
from .classifiers import DebertaV2Classifier, DebertaV2Model, DebertaV2LoRaClassifier
from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AutoModel
from transformers import AdamW, get_scheduler
import loralib as lora
from sklearn.model_selection import StratifiedKFold

class PrepareDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, additional_features=None, max_length=128):
        """
        PrepareDataset for classification tasks.
        
        Args:
            texts: List of input texts.
            labels: Labels for classification.
            tokenizer: Tokenizer for text input.
            additional_features: Additional numerical features (e.g., year, month, etc.).
            max_length: Maximum token length for text inputs.
        """
        print(f"[DATASET] Preparing dataset with {len(texts)} texts")
        start_time = time.time()

        if not isinstance(texts, list):
            texts = texts.tolist()

        # Tokenize input texts
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )

        # Labels
        self.labels = torch.tensor(labels, dtype=torch.long)

        # Normalize additional features (if provided)
        if additional_features is None:
            self.additional_features = torch.tensor([], dtype=torch.float32)
        else:
            self.additional_features = torch.tensor(additional_features, dtype=torch.float32)
            means = self.additional_features.mean(dim=0, keepdim=True)
            stds = self.additional_features.std(dim=0, keepdim=True)
            self.additional_features = (self.additional_features - means) / (stds + 1e-8)

        end_time = time.time()
        print(f"[DATASET] Dataset preparation completed in {end_time - start_time:.2f} seconds")

    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'additional_features': self.additional_features[idx] if self.additional_features.size(0) > 0 else torch.tensor([]),
            'labels': self.labels[idx]
        }
        return item

    def __len__(self):
        return len(self.labels)

class Trainer:
    def __init__(self, model):
        """
        Trainer for classification tasks.
        
        Args:
            model: An instance of the classification model.
        """
        print(f"[INIT] Initializing Trainer")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INIT] Using device: {self.device}")

        self.model = model.to(self.device)
        print("[INIT] Initialization complete")

    def validate_model(self, val_loader):
        """
        Validate the model on the validation set and return metrics.
        """
        print(f"[VALIDATION] Evaluating model on validation set...")
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    additional_features=additional_features
                )
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        report = classification_report(all_labels, all_preds, zero_division=0, output_dict=True)
        print(f"[VALIDATION] Classification Report:\n{classification_report(all_labels, all_preds, zero_division=0)}")
        return report

    def train_model(self, train_loader, val_loader, epochs=5, lr=2e-5):
        """
        Train the classification model.
        """
        print(f"[TRAINING] Starting training")
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        for epoch in range(epochs):
            print(f"[TRAINING] Epoch {epoch+1}/{epochs}")
            self.model.train()
            total_loss = 0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    additional_features=additional_features,
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            print(f"[TRAINING] Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

            # Validation
            val_metrics = self.validate_model(val_loader)
            print(f"[TRAINING] Validation Metrics (Epoch {epoch+1}): {val_metrics}")

        print("[TRAINING] Training completed!")