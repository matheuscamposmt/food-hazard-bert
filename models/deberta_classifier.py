import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config
from transformers.modeling_outputs import SequenceClassifierOutput

class DebertaV2Classifier(nn.Module):
    def __init__(self, num_classes, additional_feature_size, class_weights=None):
        super(DebertaV2Classifier, self).__init__()

        if class_weights is not None:
            if len(class_weights) != num_classes:
                raise ValueError("Number of class weights should be equal to the number of classes")
            
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
        
        self.class_weights = class_weights
        self.num_classes = num_classes
        self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-small')
        self.config = DebertaV2Config.from_pretrained('microsoft/deberta-v3-small')
        
        # Adjust the classifier input to include additional features
        self.classifier = nn.Linear(self.config.hidden_size + additional_feature_size, num_classes)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

    def forward(self, input_ids, attention_mask, additional_features, labels=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        
        # [CLS] token or pooled representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)

        # Concatenate Transformer output with external features
        pooled_output = torch.cat([pooled_output, additional_features], dim=1)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

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

import loralib as lora
class DebertaV2LoRaClassifier(DebertaV2Classifier):
    """
    A wrapper class to combine DebertaV2Classifier with LoRA.
    """

    def __init__(self, num_classes, additional_feature_size, class_weights=None):
        super(DebertaV2LoRaClassifier, self).__init__(num_classes, additional_feature_size, class_weights)
        self.classifier = lora.Linear(
            self.config.hidden_size + additional_feature_size,
            num_classes,
            r=16,        # Low-rank dimension
            lora_alpha=32,   # Scaling factor
            lora_dropout=0.1 # LoRA-specific dropout
        )

        lora.mark_only_lora_as_trainable(self)

    def save_pretrained(self, path):
        self.deberta.save_pretrained(path)
        self.config.save_pretrained(path)
        torch.save({
            'lora_state_dict': self.lora.state_dict(),
            'num_classes': self.num_classes
        }, f"{path}/custom_head.pt")




class HierarchicalClassifier(nn.Module):
    """
    A wrapper class to combine DebertaV2Classifier with LoRA for sequence classification hierarchically, i.e., first one label is predicted and then another label is predicted based on the first label.
    """

    def __init__(self, num_classes, additional_feature_size, num_classes_2):
        super(HierarchicalClassifier, self).__init__()
        self.num_classes = num_classes
        self.num_classes_2 = num_classes_2
        self.deberta = DebertaV2Model.from_pretrained('microsoft/deberta-v3-small')
        self.config = DebertaV2Config.from_pretrained('microsoft/deberta-v3-small')
        self.lora = lora.Linear(
            self.config.hidden_size + additional_feature_size,
            num_classes,
            r=16,        # Low-rank dimension
            lora_alpha=32,   # Scaling factor
            lora_dropout=0.1 # LoRA-specific dropout
        )
        self.lora_2 = lora.Linear(
            self.config.hidden_size + additional_feature_size + num_classes,
            num_classes_2,
            r=16,        # Low-rank dimension
            lora_alpha=32,   # Scaling factor
            lora_dropout=0.1 # LoRA-specific dropout
        )

    def forward(self, input_ids, attention_mask, additional_features, labels=None):
        outputs = self.deberta(input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = torch.cat([pooled_output, additional_features], dim=1)

        logits = self.lora(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))

        if self.training:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        # Inference
        predicted_labels = logits.argmax(dim=1)
        pooled_output = torch.cat([pooled_output, predicted_labels], dim=1)
        logits_2 = self.lora_2(pooled_output)
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits_2,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def save_pretrained(self, path):
        self.deberta.save_pretrained(path)
        self.config.save_pretrained(path)
        torch.save({
            'lora_state_dict': self.lora.state_dict(),
            'lora_2_state_dict': self.lora_2.state_dict(),
            'num_classes': self.num_classes,
            'num_classes_2': self.num_classes_2
        }, f"{path}/custom_head.pt")

