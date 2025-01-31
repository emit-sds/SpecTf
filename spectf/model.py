import torch
import torch.nn as nn

class BandConcat(nn.Module):
    """Module to concatenate band wavelength information to spectra.

    This serves as the positional encoding for the transformer, and replaces the
    traditional additive sinusoidal encoding. Band wavelengths are normalized to
    a fixed mean and standard deviation for stable training.
    Default mean (1440) and std (600) are set based on the EMIT spectral range.

    Attributes:
        banddef (torch.Tensor): Band center wavelengths.
        mean (int): Predefined mean of the band center wavelengths.
        std (int): Predefined stddev of the band center wavelengths.
    """

    def __init__(self, banddef: torch.Tensor, mean: int = 1440, std: int = 600):
        """Initialize BandConcat module.

        Args:
            banddef (torch.Tensor): Band center wavelengths.
            mean (int): Mean of the band center wavelengths. Default 1440.
            std (int): Stddev of the band center wavelengths. Default 600.
        """
        super().__init__()
        self.mean = mean
        self.std = std

        # Reshape from (s,) to (1, s, 1)
        self.banddef = banddef.unsqueeze(-1).unsqueeze(0)

        # Normalize band wavelengths
        self.banddef = (self.banddef - self.mean) / self.std

    def forward(self, spectra: torch.Tensor):
        """BandConcat forward pass.

        Args:
            spectra (torch.Tensor): tensor of shape (b, s, 1)

        Returns:
            torch.Tensor: concatenated tensor of shape (b, s, 2)
        """

        encoded = torch.cat((spectra, self.banddef.expand_as(spectra)), dim=-1)
        return encoded

class SpectralEmbed(nn.Module):
    """Module to embed spectra per-band using a linear layer.

    This module is used to embed each wavelength in the spectra into a 
    higher-dimensional space. This is a simple learned representation that
    replaces the traditional tokenization in transformers.
    
    Attributes:
        linear (nn.Linear): Linear layer for embedding.
        activation (nn.Module): Activation function for the embedding.
    """

    def __init__(self, n_filters: int = 64, activation: str = 'tanh'):
        """Initialize SpectralEmbed module.

        The default activation 'tanh' is recommended to mitigate exploding
        gradients during training. Other methods such as gradient clipping
        are also effective, in which case other unbounded activations like
        'relu' or 'gelu', or None at all, can be used.

        Args:
            n_filters (int): Number of filters for the linear layer. Default 64.
            activation (str): Activation function for the embedding. Default 'tanh'.
        """
        super().__init__()
        self.linear = nn.Linear(2, n_filters)
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f'SpectralEmbed activation {activation} is not implemented.')
    
    def forward(self, x: torch.Tensor):
        x = self.linear(x)
        x = self.activation(x)
        return x

class FeedForwardBlock(nn.Module):
    """Feed-forward module for transformer layers.

    This module consists of two linear layers with a GELU activation in between.
    The residual connection is disabled by default, since SpecTf is a shallow
    model that only uses a single transformer module. Deeper modules may require
    residual connections due to the vanishing gradient problem.
    
    Attributes and Architecture:
        self.linear1: Linear layer (dim_model -> dim_ff)
        self.gelu: GELU activation
        self.dropout: Dropout
        self.linear2: Linear layer (dim_ff -> dim_model)
        self.dropout2: Dropout
        self.use_residual: Whether to use residual connections.
    """

    def __init__(self, dim_model: int, dim_ff: int, dropout: float = 0.1, use_residual: bool = False):
        """ Initialize FeedForwardBlock module.

        Args:
            dim_model (int): Dimension of the input and output tensors.
            dim_ff (int): Dimension of the intermediate tensor.
            dropout (float): Dropout rate. Default 0.1.
            use_residual (bool): Whether to use residual connections. Default False.
        """

        super().__init__()
        self.linear1 = nn.Linear(dim_model, dim_ff, bias=True)
        self.linear2 = nn.Linear(dim_ff, dim_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.gelu = torch.nn.functional.gelu
        self.use_residual = use_residual

    def forward(self, x: torch.Tensor):
        """FeedForwardBlock forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b, s, dim_model)
        
        Returns:
            torch.Tensor: Output tensor of shape (b, s, dim_model)
        """
        residual = x
        x = self.linear2(self.dropout(self.gelu(self.linear1(x))))
        x = self.dropout2(x)
        if self.use_residual:
            x = x + residual
        return x

class AttentionBlock(nn.Module):
    """Attention module for transformer layers.

    This module consists of a multi-head self-attention layer with a dropout
    layer. It is based on the original PyTorch implementation. The residual
    connection is disabled by default, since SpecTf is a shallow model that
    only uses a single transformer module. Deeper modules may require residual
    connections due to the vanishing gradient problem.

    Attributes and Architecture:
        self.attention: MultiheadAttention layer
        self.dropout: Dropout
        self.use_residual: Whether to use residual connections
    """

    def __init__(self, dim_model: int, num_heads: int, dropout: float = 0.1, use_residual: bool = False):
        """ Initialize AttentionBlock module.

        Args:
            dim_model (int): Dimension of the input and output tensors.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate. Default 0.1.
            use_residual (bool): Whether to use residual connections. Default False.
        """

        super().__init__()
        self.attention = nn.MultiheadAttention(dim_model, num_heads, dropout=dropout, bias=True, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """AttentionBlock forward pass.

        Args:
            query (torch.Tensor): Query tensor of shape (b, s, dim_model)
            key (torch.Tensor): Key tensor of shape (b, s, dim_model)
            value (torch.Tensor): Value tensor of shape (b, s, dim_model)
        
        Returns:
            torch.Tensor: Output tensor of shape (b, s, dim_model)
        """
        residual = query
        x = self.attention(query, key, value)[0]
        x = self.dropout(x)
        if self.use_residual:
            x = x + residual
        return x

class EncoderLayer(nn.Module):
    """Encoder layer for transformer models.
    
    This module consists of a self-attention block followed by a feed-forward
    block. It is based on the original PyTorch implementation, and uses the
    pre-layer normalization variant per Xiong et al. (2020). The residual
    connection is disabled by default, since SpecTf is a shallow model that
    only uses a single transformer module. Deeper modules may require residual
    connections due to the vanishing gradient problem.

    Attributes and Architecture:
        norm1: LayerNorm module
        attention: AttentionBlock module
        norm2: LayerNorm module
        ff: FeedForwardBlock module
    """

    def __init__(self, dim_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1, use_residual: bool = False):
        """ Initialize EncoderLayer module.

        Args:
            dim_model (int): Dimension of the input and output tensors.
            num_heads (int): Number of attention heads. Must be a divisor of dim_model.
            dim_ff (int): Dimension of the intermediate tensor.
            dropout (float): Dropout rate. Default 0.1.
            use_residual (bool): Whether to use residual connections. Default False.
        """

        super().__init__()
        self.attention = AttentionBlock(dim_model, num_heads, dropout, use_residual)
        self.ff = FeedForwardBlock(dim_model, dim_ff, dropout, use_residual)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
    
    def forward(self, x: torch.Tensor):
        """EncoderLayer forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b, s, dim_model)
        
        Returns:
            torch.Tensor: Output tensor of shape (b, s, dim_model)
        """
        x = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = self.ff(self.norm2(x))
        return x

class DecoderLayer(nn.Module):
    """Decoder layer for transformer models.
    
    This module consists of a self-attention block, a cross-attention block,
    and a feed-forward block. It is based on the original PyTorch implementation,
    and uses the pre-layer normalization variant per Xiong et al. (2020). The
    residual connection is disabled by default, since SpecTf is a shallow model
    that only uses a single transformer module. Deeper modules may require
    residual connections due to the vanishing gradient problem.

    Attributes and Architecture:
        norm1: LayerNorm module
        self_attn: AttentionBlock module
        norm2: LayerNorm module
        cross_attn: AttentionBlock module
        norm3: LayerNorm module
        ff: FeedForwardBlock module
    """

    def __init__(self, dim_model: int, num_heads: int, dim_ff: int, dropout: float = 0.1, use_residual: bool = False):
        """ Initialize DecoderLayer module.
        
        Args:
            dim_model (int): Dimension of the input and output tensors.
            num_heads (int): Number of attention heads. Must be a divisor of dim_model.
            dim_ff (int): Dimension of the intermediate tensor.
            dropout (float): Dropout rate. Default 0.1.
            use_residual (bool): Whether to use residual connections. Default False.
        """

        super().__init__()
        self.self_attn = AttentionBlock(dim_model, num_heads, dropout, use_residual)
        self.cross_attn = AttentionBlock(dim_model, num_heads, dropout, use_residual)
        self.ff = FeedForwardBlock(dim_model, dim_ff, dropout, use_residual)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)
    
    def forward(self, x, enc_out):
        """DecoderLayer forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b, s, dim_model)
            enc_out (torch.Tensor): Encoder output tensor of shape (b, s, dim_model)
        
        Returns:
            torch.Tensor: Output tensor of shape (b, s, dim_model)
        """

        x = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = self.cross_attn(self.norm2(x), enc_out, enc_out)
        x = self.ff(self.norm3(x))
        return x
    
class SpecTfEncoder(nn.Module):
    """Encoder based Spectral Transformer model.
    
    This is the simplest Spectral Transformer architecture, consisting only of
    encoder layers. For most tasks, a single encoder layer is a good starting
    point. Aside from hyperparameters that affect the dimensionality of layers,
    the aggregation method is the most significant parameter to tune. Some
    intuition suggests that 'max' aggregation is appropriate for tasks where
    a few bands are more important than others, while 'mean' aggregation is
    appropriate when the overall shape of the spectrum is important. While the
    'flat' aggregation avoids pooling altogether, it fixes the length of the
    input sequence, and limits the model to a fixed set of bands.

    Attributes:
        band_concat: BandConcat module
        spectral_embed: SpectralEmbed module
        layers: List of EncoderLayer modules
        aggregate: Aggregation method ('mean', 'max', 'flat')
        head: Linear layer for classification or regression
    """
    def __init__(self,
                 banddef: torch.Tensor,
                 num_classes: int = 2,
                 num_heads: int = 8,
                 dim_proj: int = 64,
                 dim_ff: int = 64,
                 dropout: float = 0.1,
                 agg: str = 'max',
                 use_residual: bool = False,
                 num_layers: int = 1):
        """Initialize SpecTfEncoder module.

        Args:
            banddef (torch.Tensor): Band center wavelengths.
            num_classes (int): Number of classes for classification. Default 2.
            num_heads (int): Number of attention heads. Must be a divisor of
                             dim_proj. Default 8.
            dim_proj (int): Dimension of the projected tensors. Default 64.
            dim_ff (int): Dimension of the intermediate tensors. Default 64.
            dropout (float): Dropout rate. Default 0.1.
            agg (str): Aggregation method ('mean', 'max', 'flat').
                       Default 'max'.
            use_residual (bool): Whether to use residual connections.
                                 Default False.
            num_layers (int): Number of encoder layers. Default 1.
        """
        super().__init__()

        # Embedding
        self.band_concat = BandConcat(banddef)
        self.spectral_embed = SpectralEmbed(n_filters=dim_proj)

        # Attention
        self.layers = nn.ModuleList([
            EncoderLayer(dim_proj, num_heads, dim_ff, dropout, use_residual)
            for _ in range(num_layers)
        ])

        # Head
        if agg == 'mean':
            self.aggregate = lambda x: torch.mean(x, dim=1)
            self.head = nn.Linear(dim_proj, num_classes)
        elif agg == 'max':
            self.aggregate = lambda x: torch.max(x, dim=1)[0]
            self.head = nn.Linear(dim_proj, num_classes)
        elif agg == 'flat':
            self.aggregate = lambda x: torch.flatten(x, start_dim=1)
            self.head = nn.Linear(banddef.shape[0] * dim_proj, num_classes)
        else:
            raise ValueError(f'Aggregation method {agg} is not implemented.')

        self.initialize_weights()

    def forward(self, x: torch.Tensor):
        """SpecTfEncoder forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b, s, 1)
        
        Returns:
            torch.Tensor: Output tensor of shape (b, num_classes)
        """
        x = self.band_concat(x)
        x = self.spectral_embed(x)

        for layer in self.layers:
            x = layer(x)

        x = self.aggregate(x)
        x = self.head(x)

        return x

    def initialize_weights(self):
        """Initialize weights for the model."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

# Original implementation
# class SimpleSeqClassifier(nn.Module):
#     def __init__(self, 
#                  banddef,
#                  num_classes: int = 2,
#                  num_heads: int = 8,
#                  dim_proj: int = 64,
#                  dim_ff: int = 64,
#                  dropout: float = 0.1,
#                  agg: str = 'max'):
#         super().__init__()

#         # Embedding
#         self.band_concat = BandConcat(banddef)
#         self.spectral_embed = SpectralEmbed(n_filters=dim_proj)

#         # Attention
#         self.self_attn = nn.MultiheadAttention(dim_proj, num_heads, dropout=dropout, bias=True, batch_first=True)

#         # Feedforward
#         self.linear1 = nn.Linear(dim_proj, dim_ff, bias=True)
#         self.dropout = nn.Dropout(dropout)
#         self.linear2 = nn.Linear(dim_ff, dim_proj, bias=True)

#         # Normalization
#         self.norm1 = nn.LayerNorm(dim_proj, eps=1e-5, bias=True)
#         self.norm2 = nn.LayerNorm(dim_proj, eps=1e-5, bias=True)
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.gelu = torch.nn.functional.gelu

#         # Classification
#         self.aggregate = agg
#         self.classifier = nn.Linear(dim_proj, num_classes)
#         self.initialize_weights()


#     def forward(self, x):
#         x = self.band_concat(x)
#         x = self.spectral_embed(x)

#         # Transformer without skip connections
#         x = self._sa_block(self.norm1(x))
#         x = self._ff_block(self.norm2(x))

#         if self.aggregate == 'mean':
#             x = torch.mean(x, dim=1)
#         elif self.aggregate == 'max':
#             x,_ = torch.max(x, dim=1)
#         x = self.classifier(x)
#         return x

#     def initialize_weights(self):
#         for module in self.modules():
#             if isinstance(module, (nn.Linear, nn.Conv1d)):
#                 nn.init.xavier_uniform_(module.weight)
#                 if module.bias is not None:
#                     nn.init.constant_(module.bias, 0)

#     def _sa_block(self, x):
#         x = self.self_attn(x, x, x)[0]
#         return self.dropout1(x)
    
#     def _ff_block(self, x):
#         x = self.linear2(self.dropout(self.gelu(self.linear1(x))))
#         return self.dropout2(x)