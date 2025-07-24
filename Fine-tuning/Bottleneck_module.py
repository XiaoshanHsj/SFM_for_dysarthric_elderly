from torch import nn

class ChangeFrameAndDim(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.add_frame_layer = nn.Sequential()
        self.add_frame_layer.add_module("de_cnn",
                                        nn.ConvTranspose1d(config.hidden_size, config.hidden_size, 2, stride=2, padding=0))
        self.add_frame_layer.add_module("relu", nn.ReLU())
        # decrease the hidden_layer_dim, input: (batch_size, hidden_layer_dim)
        self.decrease_dim_layer = nn.Sequential()
        self.decrease_dim_layer.add_module("linear", nn.Linear(config.hidden_size, self.dim))
        self.decrease_dim_layer.add_module("relu", nn.ReLU())
        self.decrease_dim_layer.add_module('dropout', nn.Dropout(config.hidden_dropout))
        self.return_frame_layer = nn.Sequential()
        self.return_frame_layer.add_module("cnn",
                                        nn.Conv1d(self.dim, self.dim, 2, stride=2, padding=0))
        self.return_frame_layer.add_module("relu", nn.ReLU())
        self.return_dim_layer = nn.Sequential()
        self.return_dim_layer.add_module("linear", nn.Linear(self.dim, config.hidden_size))
        self.return_dim_layer.add_module("relu", nn.ReLU())
        self.return_dim_layer.add_module('dropout', nn.Dropout(config.hidden_dropout))


    def forward(self, hidden_state):
        # (16, 244, 1024)
        hidden_state = hidden_state.permute(0, 2, 1)
        # (16, 1024, 244)
        decnn_x = self.add_frame_layer(hidden_state)
        # (16, 1024, 488)
        decnn_x = decnn_x.permute(0, 2, 1)
        # (16, 488, 1024)
        dec_dim_x = self.decrease_dim_layer(decnn_x)
        # (16, 488, 512)
        dec_dim_x = dec_dim_x.permute(0, 2, 1)
        # (16, 512, 488)
        cnn_x = self.return_frame_layer(dec_dim_x)
        # (16, 512, 244)
        cnn_x = cnn_x.permute(0, 2, 1)
        # (16, 244, 512)
        return_x = self.return_dim_layer(cnn_x)
        # (16, 244, 1024)
        return dec_dim_x, return_x