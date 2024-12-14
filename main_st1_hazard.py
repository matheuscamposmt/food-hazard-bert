import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import DistilBertTokenizer
from training.utils import pretty_print_report
from training.st_trainer import PrepareDataset, Trainer
from training.classifiers import ST1Classifier
import json
import datetime
import torch
import os

def train():
    # Load Dataset
    df = pd.read_csv('./data/incidents_train.csv')
    df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)
    assert df.id.is_unique

    # Prepare Labels
    le_hazard_category = LabelEncoder()
    le_product_category = LabelEncoder()

    # Encode labels
    hazard_category_labels = le_hazard_category.fit_transform(df['hazard-category'])
    product_category_labels = le_product_category.fit_transform(df['product-category'])

    # Prepare Additional Features
    temporal_features = df[['year', 'month']]
    country_feature = pd.get_dummies(df['country']).astype(float).to_numpy()
    additional_features = np.concatenate([temporal_features.to_numpy(), country_feature], axis=1)

    # Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Split Dataset
    train_texts, val_texts, train_st1_labels, val_st1_labels, train_additional_features, val_additional_features = train_test_split(
        df['text'], hazard_category_labels, additional_features, test_size=0.2, random_state=42
    )

    # Prepare Datasets
    train_dataset = PrepareDataset(
        texts=train_texts,
        st1_labels=train_st1_labels,
        tokenizer=tokenizer,
        additional_features=train_additional_features
    )

    val_dataset = PrepareDataset(
        texts=val_texts,
        st1_labels=val_st1_labels,
        tokenizer=tokenizer,
        additional_features=val_additional_features,
    )

    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize Models
    model_st1 = ST1Classifier(
        num_categories=len(le_hazard_category.classes_),
        additional_feature_size=additional_features.shape[1]
    )

    label_names = le_hazard_category.classes_
    # Initialize Trainer
    trainer = Trainer(model_st1, label_names=label_names)

    # Train the Model
    epochs = 15
    learning_rate = 2e-5

    reports = trainer.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=learning_rate,
        report_callback=pretty_print_report,
        use_class_weights=True
    )

    try:
        # Save Models
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        st1_model_save_path = f"./models/st1_classifier_{timestamp}"

        print(f"[INFO] Saving ST1 model to {st1_model_save_path}")
        model_st1.save_pretrained(st1_model_save_path)

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
