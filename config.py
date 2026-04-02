from transformers.configuration_utils import PretrainedConfig



class GraphBertConfig(PretrainedConfig):

    def __init__(
        self,
        residual_type = 'none',
        x_size=80,
        y_size=7,
        k=5,
        max_inti_pos_index = 7,
        max_hop_dis_index = 102,
        #Agrovoc config
        # max_wl_role_index = 8105,
        #FoodOn config
        # max_wl_role_index = 6415,
        #Augmented_agrovoc_config
        # max_wl_role_index = 30432,
        #Augmented_foodon_config
        max_wl_role_index = 16887,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_decoder=False,
        **kwargs
    ):
        super(GraphBertConfig, self).__init__(**kwargs)
        self.max_wl_role_index = max_wl_role_index
        self.max_hop_dis_index = max_hop_dis_index
        self.max_inti_pos_index = max_inti_pos_index
        self.residual_type = residual_type
        self.x_size = x_size
        self.y_size = y_size
        self.k = k
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder

