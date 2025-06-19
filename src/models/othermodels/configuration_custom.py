"""custom config for Model from Hugging Face.

"""

from transformers import PretrainedConfig


class Custom_Config(PretrainedConfig):

    model_type = "custom"

    def __init__(
            self,
            model_name = None,
            d_model = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.d_model = d_model

class CNN_Config(PretrainedConfig):

    model_type = "cnn"

    def __init__(
            self,
            vocab_size = None,
            embedding_dim = None,
            input_len = None,
            d_model = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.input_len = input_len
        self.d_model = d_model

class LSTM_Config(PretrainedConfig):
    
    model_type = "lstm"
    def __init__(
            self,
            vocab_size = None,
            embedding_dim = None,
            hidden_size = None,
            num_layers = None,
            bidirectional = None,
            d_model = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.d_model = d_model