
import pandas as pd
from training.trainer import Trainer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from training.classifiers import DebertaV2Classifier, DebertaV2LoRaClassifier

def train():
    df = pd.read_csv('./data/incidents_train.csv')
    df.rename(columns={'Unnamed: 0': 'id'}, inplace=True)

    assert df.id.is_unique == True

    # import labelencoder
    le = LabelEncoder()
    labels = le.fit_transform(df['hazard-category'])
    temporal_features = df[['year', 'month']]
    # normalize
    #temporal_features = (temporal_features - temporal_features.mean()) / temporal_features.std()
    country_feature = df['country']
    # encode one hot
    country_feature = pd.get_dummies(country_feature).astype(float).to_numpy()
    add_feats = np.concatenate([temporal_features.to_numpy(), country_feature], axis=1)
    # compute class weights

    class_names = np.unique(df['hazard-category'])
    n_classes = len(class_names)
    model = DebertaV2Classifier(n_classes, add_feats.shape[1])
    trainer = Trainer(model)

    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    reports_ = trainer.train_model(df['title'], labels, additional_features=add_feats, epochs=15, batch_size=128, unique_labels=np.unique(labels), class_names=class_names)
    #reports = clf.train_model_cv(df['title'], labels, additional_features=add_feats, epochs=15, batch_size=128, unique_labels=np.unique(labels), class_names=class_names)
    # save reports to json
    import json
    import datetime

    timestamp = datetime.datetime.now().strftime('%d%m%Y-%H%M%S')
    filepath = f'training/reports/report_{timestamp}.json'
    with open(filepath, mode='w') as fout:
        json.dump(reports_, fout)
        print(f"\n\n[INFO] Reports saved at {filepath}")

    return reports_


if __name__ == "__main__":
    train()