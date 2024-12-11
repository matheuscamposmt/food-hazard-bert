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

class FoodHazardDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        print(f"[DATASET] Preparing dataset with {len(texts)} texts")
        start_time = time.time()
        
        # Ensure texts is a list of strings
        if not isinstance(texts, list):
            texts = texts.tolist()
        
        # Tokenization with explicit list of strings
        self.encodings = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels)
        
        end_time = time.time()
        print(f"[DATASET] Dataset preparation completed in {end_time - start_time:.2f} seconds")
    
    def __getitem__(self, idx):
        item = {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }
        return item
    
    def __len__(self):
        return len(self.labels)

from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
from transformers import AdamW, get_scheduler

class Classifier:
    def __init__(self, model_name='microsoft/deberta-v3-small'):
        """
        Initialize transformer-based text classifier
        
        Args:
            categories (list): List of unique categories to classify
            model_name (str): Transformer model to use
        """
        print(f"[INIT] Initializing Transformer Classifier")
        print(f"[INIT] Model: {model_name}")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INIT] Using device: {self.device}")
        
        self.model_name = model_name
        
        # Label encoding
        self.label_encoders = {}
        self.models = {}
        
        print("[INIT] Initialization complete")

    def train_model(self, texts, labels, category, epochs=3, batch_size=16):
        """
        Train a transformer model for a specific category
        """
        print(f"\n[TRAINING] Starting training for {category}")
        print(f"[TRAINING] Total texts: {len(texts)}")
        print(f"[TRAINING] Epochs: {epochs}")
        print(f"[TRAINING] Batch size: {batch_size}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        print(f"[TRAINING] Training set size: {len(X_train)}")
        print(f"[TRAINING] Validation set size: {len(X_val)}")
        
        # Tokenizer
        print("[TRAINING] Loading tokenizer...")
        tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)
        
        # Create datasets
        print("[TRAINING] Creating datasets...")
        train_dataset = FoodHazardDataset(X_train, y_train, tokenizer)
        val_dataset = FoodHazardDataset(X_val, y_val, tokenizer)
        
        # DataLoaders
        print("[TRAINING] Creating data loaders...")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Model initialization
        num_labels = len(np.unique(labels))
        print(f"[TRAINING] Number of labels: {num_labels}")
        
        print("[TRAINING] Loading pre-trained model...")
        model = DebertaV2ForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        ).to(self.device)
        
        # Optimizer and scheduler
        print("[TRAINING] Preparing optimizer and scheduler...")
        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs
        scheduler = get_scheduler(
            "linear", 
            optimizer=optimizer, 
            num_warmup_steps=0, 
            num_training_steps=total_steps
        )
        
        # Training loop
        print("[TRAINING] Starting training loop...")
        model.train()
        for epoch in range(epochs):
            print(f"[TRAINING] Epoch {epoch+1}/{epochs}")
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                if batch_idx % 10 == 0:
                    print(f"[TRAINING] Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        return model
    
    def predict(self, texts, category):
        """
        Predict categories for new texts
        """
        print(f"\n[PREDICTION] Predicting for {category}")
        print(f"[PREDICTION] Number of texts: {len(texts)}")
        
        # Ensure model is loaded
        if category not in self.models:
            raise ValueError(f"No model trained for {category}")
        
        # Tokenizer
        print("[PREDICTION] Loading tokenizer...")
        tokenizer = DebertaV2Tokenizer.from_pretrained(self.model_name)
        
        # Prepare input
        print("[PREDICTION] Preparing input...")
        inputs = tokenizer(
            texts, 
            truncation=True, 
            padding=True, 
            max_length=128, 
            return_tensors='pt'
        ).to(self.device)
        
        # Model prediction
        print("[PREDICTION] Running model prediction...")
        model = self.models[category].to(self.device)
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
        
        # Decode predictions
        print("[PREDICTION] Decoding predictions...")
        le = self.label_encoders[category]
        pred_labels = le.inverse_transform(preds.cpu().numpy())
        
        print("[PREDICTION] Prediction complete!")
        return pred_labels, le.classes_

def main(csv_path):
    print(f"[MAIN] Starting classification process")
    print(f"[MAIN] Loading CSV from {csv_path}")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Categories to predict
    categories = ['hazard-category', 'product-category']
    
    # Initialize classifier
    print("[MAIN] Initializing Transformer Classifier")
    classifier = Classifier(categories)
    
    # Store results for each category
    category_results = {}
    
    # Train a model for each category
    for category in categories:
        print(f"\n[MAIN] Processing {category}")
        
        # Prepare data
        texts, labels, le = classifier.prepare_data(df, category)
        
        # Train and evaluate
        print(f"[MAIN] Training model for {category}")
        results = classifier.train_model(texts, labels, category)
        
        # Store results
        category_results[category] = results
        
        # Print results
        print(f"\n[RESULTS] Results for {category}")
        print(f"Accuracy: {results['accuracy']}")
        print("Classification Report:")
        print(results['classification_report'])
        
        # Example predictions
        sample_texts = [
            "Computer with potential overheating issue",
            "Kitchen appliance safety recall"
        ]
        print("\n[MAIN] Running sample predictions")
        predictions, classes = classifier.predict(sample_texts, category)
        
        print("\nSample Predictions:")
        for text, pred in zip(sample_texts, predictions):
            print(f"Text: {text}")
            print(f"Predicted {category}: {pred}")
    
    print("\n[MAIN] Classification process complete!")
    return category_results

# Run the script
if __name__ == '__main__':
    # Ensure you have the required libraries installed:
    # pip install torch transformers pandas scikit-learn
    main('incidents_train.csv')  # Replace with your CSV path