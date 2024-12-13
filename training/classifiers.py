import os
import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config, DistilBertModel, DistilBertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from typing import Optional

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets, class_weights):
        BCE_loss = nn.CrossEntropyLoss(weight=class_weights)(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss


import torch
import torch.nn as nn

class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=1).to(labels.device)

        # One-hot encode labels
        labels_one_hot = torch.eye(probs.size(1), device=labels.device)[labels]

        # Compute precision and recall
        true_positive = torch.sum(probs * labels_one_hot, dim=0)
        predicted_positive = torch.sum(probs, dim=0)
        actual_positive = torch.sum(labels_one_hot, dim=0)

        precision = true_positive / (predicted_positive + self.epsilon)
        recall = true_positive / (actual_positive + self.epsilon)

        # Compute F1
        f1 = 2 * (precision * recall) / (precision + recall + self.epsilon)

        # Take the mean over all classes (macro F1)
        f1_loss = 1 - f1.mean()
        return f1_loss



class DebertaV2Classifier(nn.Module):
    def __init__(self, num_classes, additional_feature_size):
        super(DebertaV2Classifier, self).__init__()
        
        self.num_classes = num_classes
        self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-small')
        self.config = DebertaV2Config.from_pretrained('microsoft/deberta-v3-small')
        
        # Adjust the classifier input to include additional features
        self.classifier = nn.Linear(self.config.hidden_size + additional_feature_size, num_classes)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, additional_features, labels=None, class_weights: Optional[torch.Tensor] = None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        
        # [CLS] token or pooled representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Concatenate Transformer output with external features
        pooled_output = torch.cat([pooled_output, additional_features], dim=1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = F1Loss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1).cpu())

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_pretrained(self, path):
        self.deberta.save_pretrained(path)
        self.config.save_pretrained(path)
        torch.save({
            'classifier_state_dict': self.classifier.state_dict(),
            'num_classes': self.num_classes
        }, f"{path}/custom_head.pt")

    @classmethod
    def from_pretrained(cls, path, additional_feature_size):
        config = DebertaV2Config.from_pretrained(path)
        saved_data = torch.load(f"{path}/custom_head.pt")
        num_classes = saved_data['num_classes']

        model = cls(num_classes, additional_feature_size)
        model.deberta = DebertaV2Model.from_pretrained(path)
        model.config = config
        model.classifier.load_state_dict(saved_data['classifier_state_dict'])
        return model

class ST1Classifier(nn.Module):
    def __init__(self, num_categories, additional_feature_size=0):
        super(ST1Classifier, self).__init__()

        self.num_categories = num_categories
        self.additional_feature_size = additional_feature_size

        self.bert = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
        self.config = DistilBertConfig.from_pretrained("distilbert/distilbert-base-uncased")

        self.head = nn.Linear(
            self.config.hidden_size + additional_feature_size, num_categories
        )

    def forward(self, input_ids, attention_mask, additional_features=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)

        if additional_features is not None:
            combined_features = torch.cat([pooled_output, additional_features], dim=1)
        else:
            combined_features = pooled_output

        logits = self.head(combined_features)

        return logits

    def save_pretrained(self, save_path):
        """
        Save the model's weights and configuration to the specified path.
        """
        os.makedirs(save_path, exist_ok=True)
        self.bert.save_pretrained(save_path)
        torch.save(self.head.state_dict(), os.path.join(save_path, "head.pth"))
        print(f"[INFO] ST1Classifier saved to {save_path}")


class ST2VectorPredictor(nn.Module):
    def __init__(self, num_categories, embedding_size=768):
        super(ST2VectorPredictor, self).__init__()

        self.num_categories = num_categories

        self.bert = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
        self.config = DistilBertConfig.from_pretrained("distilbert/distilbert-base-uncased")

        self.vector_head = nn.Linear(
            self.config.hidden_size + num_categories, embedding_size
        )
        self.label_embeddings = torch.tensor([])

    def forward(self, input_ids, attention_mask, logits):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state.mean(dim=1)

        vectors = self.vector_head(
            torch.cat([pooled_output, logits], dim=1)
        )

        return vectors

    def save_pretrained(self, save_path):
        """
        Save the model's weights and configuration to the specified path.
        """
        os.makedirs(save_path, exist_ok=True)
        self.bert.save_pretrained(save_path)
        torch.save(self.vector_head.state_dict(), os.path.join(save_path, "vector_head.pth"))
        torch.save(self.label_embeddings, os.path.join(save_path, "label_embeddings.pth"))
        print(f"[INFO] ST2VectorPredictor saved to {save_path}")
