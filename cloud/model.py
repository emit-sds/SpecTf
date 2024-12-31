import torch
import torch.nn as nn

class BandConcat(nn.Module):
    """Concatenate band wavelength to reflectance spectra."""

    def __init__(self, banddef):
        super().__init__()
        self.banddef = torch.unsqueeze(banddef, -1)
        self.banddef = torch.unsqueeze(self.banddef, 0)
        self.banddef = (self.banddef - 1440) / 600

    def forward(self, spectra):
        """ 
            spectra: (b, s, 1)
            banddef: (s, 1)
        """
        encoded = torch.cat((spectra, self.banddef.expand_as(spectra)), dim=-1)
        return encoded

class SpectralEmbed(nn.Module):
    """Embed spectra and bands using Conv1D"""

    def __init__(self, n_filters: int = 64):
        super().__init__()
        self.linear = nn.Linear(2, n_filters)
    
    def forward(self, x):
        x = self.linear(x)
        x = torch.tanh(x)
        return x

class SimpleSeqClassifier(nn.Module):
    def __init__(self, 
                 banddef,
                 num_classes: int = 2,
                 num_heads: int = 8,
                 dim_proj: int = 64,
                 dim_ff: int = 64,
                 dropout: float = 0.1,
                 agg: str = 'max'):
        super().__init__()

        # Embedding
        self.band_concat = BandConcat(banddef)
        self.spectral_embed = SpectralEmbed(n_filters=dim_proj)

        # Attention
        self.self_attn = nn.MultiheadAttention(dim_proj, num_heads, dropout=dropout, bias=True, batch_first=True)

        # Feedforward
        self.linear1 = nn.Linear(dim_proj, dim_ff, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_ff, dim_proj, bias=True)

        # Normalization
        self.norm1 = nn.LayerNorm(dim_proj, eps=1e-5, bias=True)
        self.norm2 = nn.LayerNorm(dim_proj, eps=1e-5, bias=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.gelu = torch.nn.functional.gelu

        # Classification
        self.aggregate = agg
        self.classifier = nn.Linear(dim_proj, num_classes)
        self.initialize_weights()


    def forward(self, x):
        x = self.band_concat(x)
        x = self.spectral_embed(x)

        # Transformer without skip connections
        x = self._sa_block(self.norm1(x))
        x = self._ff_block(self.norm2(x))

        if self.aggregate == 'mean':
            x = torch.mean(x, dim=1)
        elif self.aggregate == 'max':
            x,_ = torch.max(x, dim=1)
        x = self.classifier(x)
        return x

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _sa_block(self, x):
        x = self.self_attn(x, x, x)[0]
        return self.dropout1(x)
    
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.gelu(self.linear1(x))))
        return self.dropout2(x)