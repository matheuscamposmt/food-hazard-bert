import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
import time
from transformers import get_scheduler
from training.classifiers import F1Loss

class PrepareDataset(Dataset):
    def __init__(self, texts, st1_labels, tokenizer, additional_features=None, st2_labels=None, max_length=128):
        """
        PrepareDataset for hierarchical classification tasks.

        Args:
            texts: List of input texts.
            st1_labels: Labels for ST1 classification.
            tokenizer: Tokenizer for text input.
            additional_features: Additional numerical features (e.g., year, month, etc.).
            st2_labels: Labels for ST2 classification (optional).
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

        # ST2 Labels (if provided)
        self.st2_labels = None
        if st2_labels is not None:
            self.st2_labels = torch.tensor(st2_labels, dtype=torch.long)

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
                'st2': self.st2_labels[idx] if self.st2_labels is not None else None
            }
        }
        return item

    def __len__(self):
        return len(self.st1_labels)


class Trainer:
    def __init__(self, model_st1, model_st2):
        """
        Trainer for ST1 and ST2 classifiers.

        Args:
            model_st1: An instance of ST1Classifier.
            model_st2: An instance of ST2VectorPredictor.
        """
        print(f"[INIT] Initializing Trainer for ST1 and ST2")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INIT] Using device: {self.device}")

        self.model_st1 = model_st1.to(self.device)
        self.model_st2 = model_st2.to(self.device)
        print("[INIT] Initialization complete")


    def validate_model(self, val_loader, label_embeddings):
        """
        Validate the model on the validation set and return metrics for ST1 and ST2.
        """
        print(f"[VALIDATION] Evaluating models on validation set...")
        self.model_st1.eval()
        self.model_st2.eval()

        st1_preds, st2_preds = [], []
        st1_labels, st2_labels = [], []
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

                # ST2 Forward pass
                vectors = self.model_st2(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits=logits
                )

                # ST2 Predictions (Cosine Similarity)
                similarities = torch.nn.functional.cosine_similarity(
                    vectors.unsqueeze(1), label_embeddings.unsqueeze(0), dim=2
                )

                st2_preds.extend(torch.argmax(similarities, dim=-1).cpu().numpy())

                # ST2 Labels (if available)
                if labels["st2"] is not None:
                    st2_labels.extend(labels["st2"].cpu().numpy())

        # Generate classification reports for ST1
        print("[VALIDATION] Generating ST1 reports...")
        st1_report = classification_report(
            st1_labels, st1_preds, output_dict=True, zero_division=0
        )

        # Generate classification reports for ST2
        st2_report = None
        if st2_labels:
            print("[VALIDATION] Generating ST2 reports...")
            st2_report = classification_report(
                st2_labels, st2_preds, output_dict=True, zero_division=0
            )

        # Return metrics as a dictionary
        return {
            "st1": st1_report,
            "st2": st2_report
        }


    def train_model(self, train_loader, val_loader, label_embeddings, epochs=5, lr=2e-5, report_callback=None):
        """
        Train the ST1 and ST2 models and collect validation metrics.
        """
        print(f"[TRAINING] Starting training")
        optimizer_st1 = torch.optim.AdamW(self.model_st1.parameters(), lr=lr)
        optimizer_st2 = torch.optim.AdamW(self.model_st2.parameters(), lr=lr)

        total_steps = len(train_loader) * epochs
        scheduler_st1 = get_scheduler(
            "linear", optimizer=optimizer_st1, num_warmup_steps=0, num_training_steps=total_steps
        )
        scheduler_st2 = get_scheduler(
            "linear", optimizer=optimizer_st2, num_warmup_steps=0, num_training_steps=total_steps
        )

        all_reports = {}
        for epoch in range(epochs):
            print(f"[TRAINING] Epoch {epoch+1}/{epochs}")
            self.model_st1.train()
            self.model_st2.train()
            total_loss_st1, total_loss_st2 = 0, 0

            for batch in train_loader:
                optimizer_st1.zero_grad()
                optimizer_st2.zero_grad()

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


                loss_fct = F1Loss()
                loss_st1: torch.Tensor = loss_fct(logits, labels['st1'].to(self.device))

                total_loss_st1 += loss_st1.item()

                loss_st1.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(self.model_st1.parameters(), 1.0)
                optimizer_st1.step()
                scheduler_st1.step()

                # ST2 Training
                vectors = self.model_st2(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits=logits.detach(),
                )

                loss_fct_cosine = torch.nn.CosineEmbeddingLoss(margin=0.5)
                loss_st2: torch.Tensor = loss_fct_cosine(vectors, label_embeddings[labels['st2'].to(self.device)], 
                                                         target=torch.ones(labels['st2'].shape[0]).to(self.device))

                total_loss_st2 += loss_st2.item()

                loss_st2.backward()
                torch.nn.utils.clip_grad_norm_(self.model_st2.parameters(), 1.0)
                optimizer_st2.step()
                scheduler_st2.step()


            print(f"[TRAINING] Epoch {epoch+1} Loss ST1: {total_loss_st1 / len(train_loader):.4f}, Loss ST2: {total_loss_st2 / len(train_loader):.4f}")

            # Validation
            val_metrics = self.validate_model(val_loader, label_embeddings)

            # Save metrics
            all_reports[f"epoch_{epoch+1}"] = val_metrics

            # Optional callback
            if report_callback:
                report_callback(epoch+1, val_metrics)

        return all_reports
