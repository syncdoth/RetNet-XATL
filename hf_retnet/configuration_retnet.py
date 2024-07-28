from transformers.configuration_utils import PretrainedConfig


class RetNetConfig(PretrainedConfig):
    model_type = "retnet"
    attribute_map = {
        "hidden_size": "decoder_embed_dim",
        "intermediate_size": "decoder_ffn_embed_dim",
        "num_attention_heads": "decoder_retention_heads",
        "num_hidden_layers": "decoder_layers",
    }

    def __init__(
            self,
            vocab_size: int = 50257,
            initializer_range: float = 0.02,
            is_decoder: bool = True,
            bos_token_id: int = None,
            pad_token_id: int = None,
            eos_token_id: int = None,
            output_retentions: bool = False,
            use_cache: bool = True,
            forward_impl: str = 'parallel',
            activation_fn: str = "gelu",
            dropout: float = 0.0,  # dropout probability
            activation_dropout: float = 0.0,  # dropout probability after activation in FFN.
            drop_path_rate: float = 0.0,
            decoder_embed_dim: int = 768,  # decoder embedding dimension
            decoder_value_embed_dim: int = 1280,  # decoder value embedding dimension
            decoder_ffn_embed_dim: int = 1280,  # decoder embedding dimension for FFN
            decoder_layers: int = 12,  # num decoder layers
            decoder_retention_heads: int = 3,  # num decoder retention heads
            decoder_normalize_before: bool = True,  # apply layernorm before each decoder block
            layernorm_embedding: bool = False,  # add layernorm to embedding
            no_scale_embedding: bool = True,  # if True, dont scale embeddings
            recurrent_chunk_size: int = 512,
            use_glu: bool = True,  # use GLU instead of FFN
            z_loss_coeff: float = 0.0,  # coefficient for z loss: TODO: 1e-4
            use_lm_decay: bool = False,
            deepnorm: bool = False,
            subln: bool = True,
            use_rms_norm: bool = True,
            groupnorm_affine: bool = False,
            layernorm_eps: float = 1e-6,
            tie_word_embeddings: bool = False,
            use_bias: bool = False,
            parallel_residual: bool = False,
            rotary_percentage: float = 1.0,
            shared_attention_norm: bool = False,
            lm_head_bias: bool = False,
            retnet_stability_update: bool = False,
            retnet_bottleneck: int = None,
            hybrid_attention_layers: list[int] = None,
            **kwargs):
        self.vocab_size = vocab_size
        self.initializer_range = initializer_range
        self.output_retentions = output_retentions
        self.use_lm_decay = use_lm_decay
        self.use_glu = use_glu
        self.z_loss_coeff = z_loss_coeff
        # size related
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_value_embed_dim = decoder_value_embed_dim
        self.decoder_retention_heads = decoder_retention_heads
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.decoder_layers = decoder_layers
        # normalization related
        self.decoder_normalize_before = decoder_normalize_before
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.drop_path_rate = drop_path_rate
        self.activation_dropout = activation_dropout
        self.no_scale_embedding = no_scale_embedding
        self.layernorm_embedding = layernorm_embedding
        self.deepnorm = deepnorm
        self.subln = subln
        self.use_rms_norm = use_rms_norm
        self.layernorm_eps = layernorm_eps
        self.use_bias = use_bias
        self.groupnorm_affine = groupnorm_affine
        self.parallel_residual = parallel_residual
        # Blockwise
        self.recurrent_chunk_size = recurrent_chunk_size
        self.forward_impl = forward_impl
        # rope
        self.rotary_percentage = rotary_percentage
        # from phi
        self.shared_attention_norm = shared_attention_norm
        self.lm_head_bias = lm_head_bias
        # retnet_stability_update
        self.retnet_stability_update = retnet_stability_update
        # others
        self.retnet_bottleneck = retnet_bottleneck
        self.hybrid_attention_layers = hybrid_attention_layers

        if self.deepnorm:
            self.decoder_normalize_before = False
            self.subln = False
        if self.subln:
            self.decoder_normalize_before = True
            self.deepnorm = False

        super().__init__(is_decoder=is_decoder,
                         bos_token_id=bos_token_id,
                         pad_token_id=pad_token_id,
                         eos_token_id=eos_token_id,
                         use_cache=use_cache,
                         tie_word_embeddings=tie_word_embeddings,
                         **kwargs)

    def override(self, args):
        for hp in self.__dict__.keys():
            if getattr(args, hp, None) is not None:
                self.__dict__[hp] = getattr(args, hp, None)
