""" Defines the SpecTf architecture and its components.

Copyright 2025 California Institute of Technology
Apache License, Version 2.0

Author: Jake Lee, jake.h.lee@jpl.nasa.gov
"""

import torch
from torch import nn


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
        are also effective, in which case None can be passed as the activation.

        Args:
            n_filters (int): Number of filters for the linear layer. Default 64.
            activation (str): Activation function for the embedding. Default 'tanh'.
        """
        super().__init__()
        self.linear = nn.Linear(2, n_filters)

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation is None:
            self.activation = nn.Identity()
        else:
            raise ValueError(f'SpectralEmbed activation {activation} is not implemented.')

    def forward(self, x: torch.Tensor):
        """SpectralEmbed forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (b, s, 2)
        
        Returns:
            torch.Tensor: Output tensor of shape (b, s, n_filters)
        """
        x = self.linear(x)
        x = self.activation(x)
        return x


class FeedForwardBlock(nn.Module):
    """Feed-forward module for transformer layers.

    This module consists of two linear layers with a GELU activation in between.
    The residual connection is disabled by default.

    Attributes and Architecture:
        self.linear1: Linear layer (dim_model -> dim_ff)
        self.gelu: GELU activation
        self.dropout: Dropout
        self.linear2: Linear layer (dim_ff -> dim_model)
        self.dropout2: Dropout
        self.use_residual: Whether to use residual connections.
    """

    def __init__(self, dim_model: int, dim_ff: int, dropout: float = 0.1,
                 use_residual: bool = False):
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
        self.gelu = nn.GELU()
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

    This module consists of a multi-head attention layer with a dropout layer.
    The residual connection is disabled by default.

    Attributes and Architecture:
        self.attention: MultiheadAttention layer
        self.dropout: Dropout
        self.use_residual: Whether to use residual connections
    """

    def __init__(self, dim_model: int, num_heads: int, dropout: float = 0.1,
                 use_residual: bool = False):
        """ Initialize AttentionBlock module.

        Args:
            dim_model (int): Dimension of the input and output tensors.
            num_heads (int): Number of attention heads.
            dropout (float): Dropout rate. Default 0.1.
            use_residual (bool): Whether to use residual connections.
                                 Default False.
        """

        super().__init__()
        self.attention = nn.MultiheadAttention(dim_model,
                                               num_heads,
                                               dropout=dropout,
                                               bias=True,
                                               batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.use_residual = use_residual

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor):
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
    block, using the pre-layer normalization variant. The residual connection
    is disabled by default.

    Attributes and Architecture:
        norm1: LayerNorm module
        attention: AttentionBlock module
        norm2: LayerNorm module
        ff: FeedForwardBlock module
    """

    def __init__(self, dim_model: int, num_heads: int, dim_ff: int,
                 dropout: float = 0.1, use_residual: bool = False):
        """ Initialize EncoderLayer module.

        Args:
            dim_model (int): Dimension of the input and output tensors.
            num_heads (int): Number of attention heads. Must be a divisor of
                             dim_model.
            dim_ff (int): Dimension of the intermediate tensor.
            dropout (float): Dropout rate. Default 0.1.
            use_residual (bool): Whether to use residual connections.
                                 Default False.
        """

        super().__init__()
        self.attention = AttentionBlock(dim_model, num_heads, dropout,
                                        use_residual)
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
    and a feed-forward block, in that order, using the pre-layer normalization
    variant. The residual connection is disabled by default.

    Attributes and Architecture:
        norm1: LayerNorm module
        self_attn: AttentionBlock module
        norm2: LayerNorm module
        cross_attn: AttentionBlock module
        norm3: LayerNorm module
        ff: FeedForwardBlock module
    """

    def __init__(self, dim_model: int, num_heads: int, dim_ff: int,
                 dropout: float = 0.1, use_residual: bool = False):
        """ Initialize DecoderLayer module.
        
        Args:
            dim_model (int): Dimension of the input and output tensors.
            num_heads (int): Number of attention heads. Must be a divisor of
                             dim_model.
            dim_ff (int): Dimension of the intermediate tensor.
            dropout (float): Dropout rate. Default 0.1.
            use_residual (bool): Whether to use residual connections.
                                 Default False.
        """

        super().__init__()
        self.self_attn = AttentionBlock(dim_model, num_heads, dropout,
                                        use_residual)
        self.cross_attn = AttentionBlock(dim_model, num_heads, dropout,
                                         use_residual)
        self.ff = FeedForwardBlock(dim_model, dim_ff, dropout, use_residual)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

    def forward(self, x, enc_out):
        """DecoderLayer forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (b, s, dim_model)
            enc_out (torch.Tensor): Encoder output tensor of shape
                                    (b, s, dim_model)
        
        Returns:
            torch.Tensor: Output tensor of shape (b, s, dim_model)
        """

        x = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = self.cross_attn(self.norm2(x), enc_out, enc_out)
        x = self.ff(self.norm3(x))
        return x


class CrossDecoderLayer(nn.Module):
    """Decoder layer with cross-attention only, and no self-attention.

    This is a DecoderLayer without its self-attention block: each query
    attends into the input token sequence independently of the other queries.
    Uses the pre-layer normalization variant.

    Attributes and Architecture:
        norm2: LayerNorm module
        cross_attn: AttentionBlock module
        norm3: LayerNorm module
        ff: FeedForwardBlock module
    """

    def __init__(self, dim_model: int, num_heads: int, dim_ff: int,
                 dropout: float = 0.1, use_residual: bool = False):
        """ Initialize CrossDecoderLayer module.

        Args:
            dim_model (int): Dimension of the input and output tensors.
            num_heads (int): Number of attention heads. Must be a divisor of
                             dim_model.
            dim_ff (int): Dimension of the intermediate tensor.
            dropout (float): Dropout rate. Default 0.1.
            use_residual (bool): Whether to use residual connections.
                                 Default False.
        """

        super().__init__()
        self.cross_attn = AttentionBlock(dim_model, num_heads, dropout,
                                         use_residual)
        self.ff = FeedForwardBlock(dim_model, dim_ff, dropout, use_residual)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

    def forward(self, x, enc_out):
        """CrossDecoderLayer forward pass.

        Args:
            x (torch.Tensor): Query tensor of shape (b, q, dim_model)
            enc_out (torch.Tensor): Encoder output tensor of shape
                                    (b, s, dim_model)

        Returns:
            torch.Tensor: Output tensor of shape (b, q, dim_model)
        """
        x = self.cross_attn(self.norm2(x), enc_out, enc_out)
        x = self.ff(self.norm3(x))
        return x


class PoolDecoderLayer(nn.Module):
    """Decoder layer with cross-attention before self-attention.

    This module consists of a cross-attention block, a self-attention block,
    and a feed-forward block, in that order, using the pre-layer normalization
    variant. This is the reverse of DecoderLayer's ordering: queries pool from
    the input tokens first, so the self-attention block operates on retrieved
    features rather than on the query vectors alone.

    Unlike elsewhere in this module, the residual connection is enabled by
    default. With it disabled, the self-attention block replaces each query's
    pooled features with a mixture across queries rather than refining them.

    Attributes and Architecture:
        norm1: LayerNorm module
        cross_attn: AttentionBlock module
        norm2: LayerNorm module
        self_attn: AttentionBlock module
        norm3: LayerNorm module
        ff: FeedForwardBlock module
    """

    def __init__(self, dim_model: int, num_heads: int, dim_ff: int,
                 dropout: float = 0.1, use_residual: bool = True):
        """ Initialize PoolDecoderLayer module.

        Args:
            dim_model (int): Dimension of the input and output tensors.
            num_heads (int): Number of attention heads. Must be a divisor of
                             dim_model.
            dim_ff (int): Dimension of the intermediate tensor.
            dropout (float): Dropout rate. Default 0.1.
            use_residual (bool): Whether to use residual connections.
                                 Default True.
        """

        super().__init__()
        self.cross_attn = AttentionBlock(dim_model, num_heads, dropout,
                                         use_residual)
        self.self_attn = AttentionBlock(dim_model, num_heads, dropout,
                                        use_residual)
        self.ff = FeedForwardBlock(dim_model, dim_ff, dropout, use_residual)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.norm3 = nn.LayerNorm(dim_model)

    def forward(self, x, enc_out):
        """PoolDecoderLayer forward pass.

        Args:
            x (torch.Tensor): Query tensor of shape (b, q, dim_model)
            enc_out (torch.Tensor): Encoder output tensor of shape
                                    (b, s, dim_model)

        Returns:
            torch.Tensor: Output tensor of shape (b, q, dim_model)
        """
        x = self.cross_attn(self.norm1(x), enc_out, enc_out)
        x = self.self_attn(self.norm2(x), self.norm2(x), self.norm2(x))
        x = self.ff(self.norm3(x))
        return x


class LabelQueryPooling(nn.Module):
    """Pools a token sequence into per-class logits via learned label queries.

    Replaces shared mean/max pooling with one learned query vector per class.
    Each query is cross-attended into the input token sequence across one or
    more decoder layers, giving each class its own pooled representation
    instead of a single pooled vector shared across classes. Pooling is
    permutation- and length-invariant over the input tokens, so the model
    remains applicable across band counts and samplings.

    With use_self_attn=False, queries only cross-attend into the input tokens
    (CrossDecoderLayer). With use_self_attn=True, queries additionally
    self-attend to each other after cross-attending, so each class's pooled
    representation depends on the others (PoolDecoderLayer).

    The final per-class logit is computed with an independent weight vector
    per class rather than a single shared nn.Linear, so classes do not mix
    through shared head weights.

    Attributes:
        label_queries (nn.Parameter): Learned per-class query vectors,
            shape (n_classes, dim_proj).
        layers: List of PoolDecoderLayer (use_self_attn=True) or
            CrossDecoderLayer (use_self_attn=False) modules.
        head_weight (nn.Parameter): Per-class head weight, shape
            (n_classes, dim_proj).
        head_bias (nn.Parameter): Per-class head bias, shape (n_classes,).
    """

    def __init__(self, dim_proj: int, n_classes: int, num_heads: int,
                 dim_ff: int, dropout: float = 0.1, num_layers: int = 1,
                 use_residual: bool = True, use_self_attn: bool = False):
        """Initialize LabelQueryPooling module.

        Args:
            dim_proj (int): Dimension of the input tokens and label queries.
            n_classes (int): Number of classes (label queries/output logits).
            num_heads (int): Number of attention heads. Must be a divisor of
                             dim_proj.
            dim_ff (int): Dimension of the intermediate feed-forward tensor.
            dropout (float): Dropout rate. Default 0.1.
            num_layers (int): Number of pooling decoder layers. Default 1.
            use_residual (bool): Whether to use residual connections in the
                                 pooling layers. Default True.
            use_self_attn (bool): Whether label queries self-attend to each
                                  other after cross-attending into the input
                                  tokens. Default False.
        """
        super().__init__()
        self.use_self_attn = use_self_attn

        self.label_queries = nn.Parameter(torch.empty(n_classes, dim_proj))
        nn.init.trunc_normal_(self.label_queries, std=0.02)

        layer_cls = PoolDecoderLayer if use_self_attn else CrossDecoderLayer
        self.layers = nn.ModuleList([
            layer_cls(dim_proj, num_heads, dim_ff, dropout, use_residual)
            for _ in range(num_layers)
        ])

        self.head_weight = nn.Parameter(torch.empty(n_classes, dim_proj))
        self.head_bias = nn.Parameter(torch.zeros(n_classes))
        nn.init.xavier_uniform_(self.head_weight)

    def forward(self, tokens: torch.Tensor):
        """LabelQueryPooling forward pass.

        Args:
            tokens (torch.Tensor): Input token sequence, shape (b, s, dim_proj)

        Returns:
            torch.Tensor: Per-class logits, shape (b, n_classes)
        """
        x = self.label_queries.unsqueeze(0).expand(tokens.shape[0], -1, -1)

        for layer in self.layers:
            x = layer(x, tokens)

        return (x * self.head_weight).sum(dim=-1) + self.head_bias


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

    Model weights are initialized using Xavier initialization and model biases
    are initialized to zero with self.initialize_weights().

    Attributes:
        band_concat: BandConcat module
        spectral_embed: SpectralEmbed module
        layers: List of EncoderLayer modules
        aggregate: Aggregation method ('mean', 'max', 'flat')
        head: Linear layer for classification or regression
    """
    def __init__(self,
                 banddef: torch.Tensor,
                 dim_output: int = 2,
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
            dim_output (int): Output dimension of the model. Default 2.
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
        self.agg = agg
        if agg == 'flat':
            self.head = nn.Linear(banddef.shape[0] * dim_proj, dim_output)
        else:
            self.head = nn.Linear(dim_proj, dim_output)

        self.initialize_weights()

    def aggregate(self, x):
        """Performs the selected aggregation method. Needs to be broken out here for PyTorch's JiT"""
        if self.agg == 'mean':
            return torch.mean(x, dim=1)
        elif self.agg == 'max':
            return torch.max(x, dim=1)[0]
        elif self.agg == 'flat':
            return torch.flatten(x, start_dim=1)
        else:
            raise ValueError(f'Aggregation method {self.agg} is not implemented.')

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


class SpecTfLabelQueryEncoder(nn.Module):
    """Encoder-based Spectral Transformer model with label-query pooling.

    Identical to SpecTfEncoder, but replaces shared mean/max/flat aggregation
    and the linear head with LabelQueryPooling: one learned query vector per
    class, cross-attended into the encoder's token sequence, giving each class
    its own pooled representation instead of a single vector shared across
    classes. See LabelQueryPooling for the use_self_attn option.

    Residual connections are configured separately for the encoder layers
    (use_residual, default False) and the pooling layers (use_pool_residual,
    default True).

    Model weights are initialized using Xavier initialization and model biases
    are initialized to zero with self.initialize_weights().

    Attributes:
        band_concat: BandConcat module
        spectral_embed: SpectralEmbed module
        layers: List of EncoderLayer modules
        pool: LabelQueryPooling module
    """

    def __init__(self,
                 banddef: torch.Tensor,
                 dim_output: int = 2,
                 num_heads: int = 8,
                 dim_proj: int = 64,
                 dim_ff: int = 64,
                 dropout: float = 0.1,
                 use_residual: bool = False,
                 num_layers: int = 1,
                 num_pool_layers: int = 1,
                 use_self_attn: bool = False,
                 use_pool_residual: bool = True):
        """Initialize SpecTfLabelQueryEncoder module.

        Args:
            banddef (torch.Tensor): Band center wavelengths.
            dim_output (int): Number of classes / label queries. Default 2.
            num_heads (int): Number of attention heads. Must be a divisor of
                             dim_proj. Default 8.
            dim_proj (int): Dimension of the projected tensors. Default 64.
            dim_ff (int): Dimension of the intermediate tensors. Default 64.
            dropout (float): Dropout rate. Default 0.1.
            use_residual (bool): Whether to use residual connections in the
                                 encoder layers. Default False.
            num_layers (int): Number of encoder layers. Default 1.
            num_pool_layers (int): Number of label-query pooling layers.
                                   Default 1.
            use_self_attn (bool): Whether label queries self-attend to each
                                  other after cross-attending into the encoder
                                  tokens. Default False.
            use_pool_residual (bool): Whether to use residual connections in
                                      the pooling layers. Default True.
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

        # Label-query pooling + head
        self.pool = LabelQueryPooling(dim_proj, dim_output, num_heads, dim_ff,
                                      dropout, num_pool_layers,
                                      use_pool_residual, use_self_attn)

        self.initialize_weights()

    def forward(self, x: torch.Tensor):
        """SpecTfLabelQueryEncoder forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (b, s, 1)

        Returns:
            torch.Tensor: Output tensor of shape (b, num_classes)
        """
        x = self.band_concat(x)
        x = self.spectral_embed(x)

        for layer in self.layers:
            x = layer(x)

        return self.pool(x)

    def initialize_weights(self):
        """Initialize weights for the model.

        Note: LabelQueryPooling initializes its own label_queries and
        head_weight parameters in __init__, so only nn.Linear/nn.Conv1d
        modules (encoder layers, attention projections) are reinitialized
        here.
        """
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
