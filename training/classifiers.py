import os
import torch
import torch.nn as nn
from transformers import DistilBertModel

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states):
        scores = self.attention(hidden_states)  # (batch_size, seq_len, 1)
        weights = torch.softmax(scores, dim=1)  # (batch_size, seq_len, 1)
        weighted_output = (hidden_states * weights).sum(dim=1)  # (batch_size, hidden_size)
        return weighted_output


class ST1Classifier(nn.Module):
    def __init__(self, num_categories, additional_feature_size=0, dropout_rate=0.3, hidden_dim=128):
        super(ST1Classifier, self).__init__()

        self.num_categories = num_categories
        self.additional_feature_size = additional_feature_size

        self.bert = DistilBertModel.from_pretrained("distilbert/distilbert-base-uncased")
        self.attention = Attention(self.bert.config.hidden_size + additional_feature_size)

        # Definir camadas
        self.linear = nn.Linear(self.bert.config.hidden_size + additional_feature_size, hidden_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.head = nn.Linear(hidden_dim, num_categories)

    def forward(self, input_ids, attention_mask, additional_features=None):
        # Obter saídas do DistilBERT
        bert_outputs = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state

        # Adicionar uma dimensão às features adicionais, se necessário
        if additional_features is not None:
            # Expandir additional_features para corresponder ao comprimento da sequência de bert_outputs
            batch_size, seq_len, hidden_size = bert_outputs.shape
            additional_features_expanded = additional_features.unsqueeze(1).expand(batch_size, seq_len, -1)

            # Concatenar as features adicionais com as saídas do BERT ao longo da última dimensão
            combined_features = torch.cat([bert_outputs, additional_features_expanded], dim=-1)
        else:
            combined_features = bert_outputs

        # Aplicar o mecanismo de atenção às features combinadas
        attention_output = self.attention(combined_features)

        # Aplicar transformações
        x = self.dropout(attention_output)
        x = self.activation(self.linear(x))
        logits = self.head(x)

        return logits

    def save_pretrained(self, save_path):
        """
        Salvar os pesos e a configuração do modelo no caminho especificado.
        """
        os.makedirs(save_path, exist_ok=True)
        self.bert.save_pretrained(save_path)
        torch.save(self.state_dict(), os.path.join(save_path, "classifier_weights.pth"))
        print(f"[INFO] ST1Classifier salvo em {save_path}")

    @classmethod
    def from_pretrained(cls, pretrained_path, num_categories, additional_feature_size=0, dropout_rate=0.3, hidden_dim=128):
        """
        Carregar os pesos e a configuração do modelo a partir do caminho especificado.
        """
        # Inicializar um novo modelo
        model = cls(
            num_categories=num_categories,
            additional_feature_size=additional_feature_size,
            dropout_rate=dropout_rate,
            hidden_dim=hidden_dim,
        )
        # Carregar os pesos do DistilBERT
        model.bert = DistilBertModel.from_pretrained(pretrained_path)
        # Carregar os pesos do classificador
        classifier_weights_path = os.path.join(pretrained_path, "classifier_weights.pth")
        if os.path.exists(classifier_weights_path):
            model.load_state_dict(torch.load(classifier_weights_path, map_location=torch.device('cpu')))
            print(f"[INFO] Pesos do ST1Classifier carregados de {classifier_weights_path}")
        else:
            print(f"[WARNING] Pesos do classificador não encontrados em {classifier_weights_path}, apenas os pesos do DistilBERT foram carregados.")
        return model
