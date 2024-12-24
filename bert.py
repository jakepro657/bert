import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.d_model = d_model

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = src * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        output = self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        return output

class Bert(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(Bert, self).__init__()
        self.encoder = Transformer(d_model, nhead, num_encoder_layers, dim_feedforward, dropout)
        self.pooler = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        x = self.encoder(x, src_mask, src_key_padding_mask)
        pooled_output = self.pooler(x[:, 0])
        pooled_output = self.activation(pooled_output)
        return x, pooled_output
