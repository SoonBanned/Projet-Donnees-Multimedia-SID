import torch
import torch.nn as nn
import torch.nn.functional as F


# Video Model
class SelfAttention(nn.Module):
    """
    Self-Attention Mechanism for sequence data (LSTM outputs).
    """

    def __init__(self, hidden_dim, attention_dim, attn_dropout_rate=0.3):
        super(SelfAttention, self).__init__()
        # Bi-LSTM output is hidden_dim * 2 if num_layers=1
        self.lstm_output_dim = hidden_dim * 2

        self.W_a = nn.Linear(self.lstm_output_dim, attention_dim, bias=False)
        self.V_a = nn.Linear(attention_dim, 1, bias=False)
        self.dropout = nn.Dropout(p=attn_dropout_rate)

    def forward(self, H):
        """
        H: (batch_size, sequence_length, lstm_output_dim) -> LSTM output sequence
        """
        # Apply projection and activation
        U = torch.tanh(self.W_a(H))  # (batch_size, sequence_length, attention_dim)

        # Compute attention scores
        scores = self.V_a(U).squeeze(-1)  # (batch_size, sequence_length)

        # Compute attention weights using softmax
        weights = F.softmax(scores, dim=1)  # (batch_size, sequence_length)

        # Compute context vector (weighted sum of LSTM outputs)
        # weights.unsqueeze(-1): (batch_size, sequence_length, 1)
        # H:                   (batch_size, sequence_length, lstm_output_dim)
        # Result:              (batch_size, lstm_output_dim)
        context = torch.sum(weights.unsqueeze(-1) * H, dim=1)

        # Apply dropout to the context vector
        context = self.dropout(context)

        return context, weights  # Return weights for potential analysis


class SA_LSTM_Classification_Model(nn.Module):
    """
    Sequence-to-Classification Model using Bi-directional LSTM and Self-Attention.
    Reduced complexity version.
    """

    def __init__(
        self,
        video_feature_dim: int,
        hidden_dim: int,
        attention_dim: int,
        num_classes: int,
    ):
        super(SA_LSTM_Classification_Model, self).__init__()

        self.lstm = nn.LSTM(
            input_size=video_feature_dim,
            hidden_size=hidden_dim,
            num_layers=LSTM_LAYERS,  # Reduced layers
            batch_first=True,
            dropout=0.5 if LSTM_LAYERS > 1 else 0.0,  # Dropout only between LSTM layers
            bidirectional=True,
        )

        lstm_output_dim = hidden_dim * 2  # Since bidirectional=True

        # Layer Normalization after LSTM - often helps stabilize training
        self.norm_lstm_out = nn.LayerNorm(lstm_output_dim)

        # Self-Attention mechanism
        self.attention = SelfAttention(
            hidden_dim, attention_dim
        )  # Uses updated dimensions

        # Layer Normalization after concatenating attention context and avg pooling
        self.norm_combined = nn.LayerNorm(lstm_output_dim * 2)

        # Classifier head
        classifier_intermediate_dim = 256  # Intermediate dimension in classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  # Increased initial dropout
            nn.Linear(
                lstm_output_dim * 2, classifier_intermediate_dim
            ),  # Combined features -> Intermediate
            nn.ReLU(),
            nn.Dropout(0.5),  # Dropout before final layer
            nn.Linear(
                classifier_intermediate_dim, num_classes
            ),  # Intermediate -> Output Classes
        )

    def forward(self, features: torch.Tensor):
        # features shape: (batch_size, sequence_length, video_feature_dim)

        # LSTM layer
        # H_out shape: (batch_size, sequence_length, hidden_dim * num_directions)
        H_out, _ = self.lstm(features)

        # Apply Layer Normalization to LSTM output sequence
        H_out_norm = self.norm_lstm_out(H_out)

        # Apply Self-Attention to normalized LSTM output
        # context shape: (batch_size, hidden_dim * num_directions)
        context, _ = self.attention(H_out_norm)

        # Average Pooling over the sequence dimension
        # avg_pool shape: (batch_size, hidden_dim * num_directions)
        avg_pool = torch.mean(H_out_norm, dim=1)

        # Concatenate Attention context and Average Pooling results
        # combined shape: (batch_size, hidden_dim * num_directions * 2)
        combined = torch.cat([context, avg_pool], dim=1)

        # Apply Layer Normalization to the combined features
        combined_norm = self.norm_combined(combined)

        # Classifier to get logits
        logits = self.classifier(combined_norm)  # Use normalized combined features

        return logits


import torch
import torch.nn as nn
from transformers import DistilBertModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BERTLSTMClassifier(nn.Module):
    def __init__(self, output_dim=20, hidden_dim=256, n_layers=2, dropout=0.4, bert_dir=None):
        super().__init__()
        # ⬇️ charge le BERT finetuné depuis ton dossier (model.safetensors pris en charge)
        if bert_dir is None:
            bert_dir = "distilbert-base-uncased"
        self.bert = DistilBertModel.from_pretrained(bert_dir)
        self.lstm = nn.LSTM(768, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.caption_attention = nn.Sequential(
            nn.Linear(768, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids_list, attention_mask_list):
        video_embs = []
        for input_ids, att_mask in zip(input_ids_list, attention_mask_list):
            out = self.bert(input_ids=input_ids.to(DEVICE),
                            attention_mask=att_mask.to(DEVICE))
            cls = out.last_hidden_state[:, 0, :]          # [n_caps, 768]
            attn = torch.softmax(self.caption_attention(cls), dim=0)  # [n_caps, 1]
            weighted = (attn * cls).sum(dim=0)            # weighted mean
            video_embs.append(weighted)
        video_embs = torch.stack(video_embs)
        lstm_input = video_embs.unsqueeze(1)
        _, (h, _) = self.lstm(lstm_input)
        return self.fc(self.dropout(h[-1]))