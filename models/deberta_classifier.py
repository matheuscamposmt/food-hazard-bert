import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config
from transformers.modeling_outputs import SequenceClassifierOutput

class DebertaV2Classifier(nn.Module):
    def __init__(self, num_classes, additional_feature_size):
        super(DebertaV2Classifier, self).__init__()
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
            loss_fct = nn.CrossEntropyLoss()
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
