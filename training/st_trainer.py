import numpy as np
from sklearn.utils import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import time
from transformers import get_scheduler
from sklearn.metrics import f1_score

class PrepareDataset(Dataset):
    def __init__(self, texts, st1_labels, tokenizer, additional_features=None, max_length=128):
        """
        PrepareDataset for hierarchical classification tasks.

        Args:
            texts: List of input texts.
            st1_labels: Labels for ST1 classification.
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

        # ST1 Labels
        self.st1_labels = torch.tensor(st1_labels, dtype=torch.long)

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
            'labels': {
                'st1': self.st1_labels[idx],
            },
        }
        return item

    def __len__(self):
        return len(self.st1_labels)


class Trainer:
    def __init__(self, model_st1, label_names=None):
        """
        Trainer for ST1 classifiers.

        Args:
            model_st1: An instance of ST1Classifier.
        """
        print(f"[INIT] Initializing Trainer for ST1")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INIT] Using device: {self.device}")

        self.model_st1 = model_st1.to(self.device)
        print("[INIT] Initialization complete")

        self.label_names = label_names


    def validate_model(self, val_loader):
        """
        Validate the model on the validation set and return metrics for ST1.
        """
        print(f"[VALIDATION] Evaluating models on validation set...")
        self.model_st1.eval()

        st1_preds = []
        st1_labels = []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)
                labels = batch['labels']

                # ST1 Forward pass
                logits = self.model_st1(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    additional_features=additional_features
                )

                # ST1 Predictions
                st1_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                st1_labels.extend(labels["st1"].cpu().numpy())
                
        # Generate classification reports for ST1
        print("[VALIDATION] Generating ST1 reports...")
        st1_report = classification_report(
            st1_labels, st1_preds, output_dict=True, zero_division=0,
            target_names=self.label_names
        )
        print(f1_score(st1_labels, st1_preds, average='macro'))

        # Return metrics as a dictionary
        return {
            "st1": st1_report,
        }
    def train_model(self, train_loader, val_loader, epochs=5, lr=0.002, report_callback=None, use_class_weights=False):
        """
        Train the ST1 models and collect validation metrics.
        """
        print(f"[TRAINING] Starting training")
        optimizer_st1 = torch.optim.AdamW(self.model_st1.parameters(), lr=lr)

        total_steps = len(train_loader) * epochs
        scheduler_st1 = get_scheduler(
            "linear", optimizer=optimizer_st1, num_warmup_steps=0, num_training_steps=total_steps
        )

        # Compute class weights once, if enabled
        class_weights = None
        if use_class_weights:
            all_labels = []
            labels_names = self.label_names
            for batch in train_loader:
                all_labels.extend(batch['labels']['st1'].numpy())  # Collect all training labels
            unique_classes = np.unique(all_labels)
            class_weights_np = compute_class_weight('balanced', classes=unique_classes, y=all_labels)
            class_weights = torch.tensor(class_weights_np, dtype=torch.float).to(self.device)

            weight_dict = {}

            for name, weight in zip(labels_names, class_weights):
                weight_dict[name] = weight
                
            print(f"[INFO] Computed class weights: {weight_dict}")

        all_reports = {}
        for epoch in range(epochs):
            print(f"[TRAINING] Epoch {epoch+1}/{epochs}")
            self.model_st1.train()
            total_loss_st1 = 0

            for batch in train_loader:
                optimizer_st1.zero_grad()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                additional_features = batch['additional_features'].to(self.device)
                labels = batch['labels']

                # ST1 Training
                logits = self.model_st1(
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    additional_features=additional_features
                )

                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
                loss_st1: torch.Tensor = loss_fct(logits, labels['st1'].to(self.device))

                total_loss_st1 += loss_st1.item()

                loss_st1.backward()
                torch.nn.utils.clip_grad_norm_(self.model_st1.parameters(), 1.0)
                optimizer_st1.step()
                scheduler_st1.step()

            print(f"[TRAINING] Epoch {epoch+1} Loss ST1: {total_loss_st1 / len(train_loader):.4f}")

            # Validation
            val_metrics = self.validate_model(val_loader)

            # Save metrics
            all_reports[f"epoch_{epoch+1}"] = val_metrics

            # Optional callback
            if report_callback:
                report_callback(epoch+1, val_metrics)

        return all_reports
