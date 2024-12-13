import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer
from training.utils import pretty_print_report
from training.st_trainer import PrepareDataset, Trainer
from training.classifiers import ST1Classifier, ST2VectorPredictor
import json
import datetime
import torch
import os


class LabelDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )


def compute_label_embeddings_in_batches(model, texts, tokenizer, batch_size=32):
    """
    Compute label embeddings in batches.

    Args:
        model: The backbone model to use for embedding computation.
        texts: List of texts representing each label.
        tokenizer: Tokenizer for encoding texts.
        batch_size: Number of samples per batch.

    Returns:
        Tensor of label embeddings of size (num_labels, hidden_size).
    """
    model.eval()
    device = next(model.parameters()).device

    dataset = LabelDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            outputs = model.bert(input_ids=input_ids, attention_mask=attention_mask)
            mean_embeddings = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(mean_embeddings.cpu())

    return torch.cat(embeddings, dim=0)


def train():
    # Load Dataset
    df = pd.read_csv('./data/incidents_train.csv')
    df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    assert df.id.is_unique

    # Prepare Labels for ST1 and ST2
    le_hazard_category = LabelEncoder()
    le_product_category = LabelEncoder()
    le_hazard = LabelEncoder()
    le_product = LabelEncoder()

    # Encode ST1 labels
    hazard_category_labels = le_hazard_category.fit_transform(df['hazard-category'])
    product_category_labels = le_product_category.fit_transform(df['product-category'])

    # Encode ST2 labels (if available)
    st2_labels = None
    if 'hazard' in df.columns and 'product' in df.columns:
        hazard_labels = le_hazard.fit_transform(df['hazard'])
        product_labels = le_product.fit_transform(df['product'])
        st2_labels = hazard_labels

    # Prepare Additional Features
    temporal_features = df[['year', 'month']]
    country_feature = pd.get_dummies(df['country']).astype(float).to_numpy()
    additional_features = np.concatenate([temporal_features.to_numpy(), country_feature], axis=1)

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Check for Precomputed Label Embeddings
    hazard_embeddings_path = './data/hazard_embeddings.pt'
    product_embeddings_path = './data/product_embeddings.pt'

    if os.path.exists(hazard_embeddings_path):
        print("[INFO] Loading precomputed label embeddings from disk")
        hazard_embeddings = torch.load(hazard_embeddings_path)
        #product_embeddings = torch.load(product_embeddings_path)
    else:
        print("[INFO] Computing label embeddings...")
        hazard_texts = le_hazard.inverse_transform(range(len(le_hazard.classes_)))
        product_texts = le_product.inverse_transform(range(len(le_product.classes_)))

        model_st1_temp = ST1Classifier(
            num_categories=len(le_hazard_category.classes_),
            additional_feature_size=additional_features.shape[1]
        )

        hazard_embeddings = compute_label_embeddings_in_batches(model_st1_temp, hazard_texts, tokenizer, batch_size=64)
        #product_embeddings = compute_label_embeddings_in_batches(model_st1_temp, product_texts, tokenizer, batch_size=64)

        # Save embeddings to disk
        torch.save(hazard_embeddings, hazard_embeddings_path)
        #torch.save(product_embeddings, product_embeddings_path)
        print(f"[INFO] Saved label embeddings to {hazard_embeddings_path} and {product_embeddings_path}")

    # Split Dataset
    train_texts, val_texts, train_st1_labels, val_st1_labels, train_additional_features, val_additional_features = train_test_split(
        df['text'], hazard_category_labels, additional_features, test_size=0.2, random_state=42
    )

    train_st2_labels, val_st2_labels = None, None
    if st2_labels is not None:
        train_st2_labels, val_st2_labels = train_test_split(st2_labels, test_size=0.2, random_state=42)

    # Prepare Datasets
    train_dataset = PrepareDataset(
        texts=train_texts,
        st1_labels=train_st1_labels,
        tokenizer=tokenizer,
        additional_features=train_additional_features,
        st2_labels=train_st2_labels
    )

    val_dataset = PrepareDataset(
        texts=val_texts,
        st1_labels=val_st1_labels,
        tokenizer=tokenizer,
        additional_features=val_additional_features,
        st2_labels=val_st2_labels
    )

    # DataLoaders
    batch_size = 96
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Models
    model_st1 = ST1Classifier(
        num_categories=len(le_hazard_category.classes_),
        additional_feature_size=additional_features.shape[1]
    )

    model_st2 = ST2VectorPredictor(
        num_categories=len(le_hazard_category.classes_),
        embedding_size=hazard_embeddings.size(1)
    )

    label_embeddings_hazard = hazard_embeddings.to('cuda:0')
    #label_embeddings_product = product_embeddings.to('cuda:0')

    # Initialize Trainer
    trainer = Trainer(model_st1, model_st2)

    # Train the Model
    epochs = 15
    learning_rate = 2e-5

    reports = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        label_embeddings=label_embeddings_hazard,
        epochs=epochs,
        lr=learning_rate,
        report_callback=pretty_print_report
    )

    try:
        # Save Models
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        st1_model_save_path = f"./models/st1_classifier_{timestamp}"
        st2_model_save_path = f"./models/st2_vector_predictor_{timestamp}"

        print(f"[INFO] Saving ST1 model to {st1_model_save_path}")
        model_st1.save_pretrained(st1_model_save_path)

        print(f"[INFO] Saving ST2 model to {st2_model_save_path}")
        torch.save(model_st2.state_dict(), st2_model_save_path)

    except Exception as e:
        print("Failed to save the models", e)

    try:
        # Save Reports
        reports_path = f"training/reports/report_{timestamp}.json"
        print(f"[INFO] Saving training reports to {reports_path}")
        with open(reports_path, "w") as report_file:
            json.dump(reports, report_file)

    except Exception as e:
        print("Failed to save the reports", e)

    return reports

if __name__ == "__main__":
    train()
