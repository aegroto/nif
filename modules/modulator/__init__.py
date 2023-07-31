from torch import nn

from modules.modulator.utils import build_dumper
from modules.positional_encoder import PositionalEncoder
from modules.siren.utils import Sine, build_linear_initializer, initialize_first_layer

class Modulator(nn.Module):
    def __init__(self, 
        input_features,
        hidden_sizes,
        encoder_params,
        omega=30.0,
        sigma=10.0,
        output_features=1,
        writer=None
    ):
        super(Modulator, self).__init__()

        self.omega = omega
        self.sigma = sigma
        self.output_features = output_features

        self.activation = Sine(1.0)
        self.final_activation = nn.Identity()


        self.encoder = PositionalEncoder(input_features, **encoder_params)

        self.head = nn.Linear(self.encoder.out_dim, hidden_sizes[0])

        self.hidden_layers = nn.ModuleList()
        for i in range(0, len(hidden_sizes)-1):
            self.hidden_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        self.tail = nn.Linear(hidden_sizes[-1], self.output_features)

        self.__dumper = build_dumper(writer)

    def initialize_weights(self):
        self.head.apply(initialize_first_layer)
        initializer = build_linear_initializer(6.0, self.omega)
        self.hidden_layers.apply(initializer)
        self.tail.apply(initializer)

    def group_features(self, y):
        features = y.size(-1)
        y = y.unflatten(-1, (features // self.output_features, self.output_features))
        return y

    def ungroup_features(self, y):
        return y.flatten(-2, -1)

    def forward(self, x):
        x = self.encoder(x)

        y = self.head(x)
        y = self.activation(self.omega * y)

        for layer in self.hidden_layers:
            y = layer(y)
            y = self.activation(self.omega * y)

        y = self.tail(y)
        y = self.final_activation(y)

        y = self.sigma * y

        if self.__dumper:
            self.__dumper(y.movedim(-1, 0))
            self.__dumper.dump_scalar("abs_mean", y.abs().mean())

        y = y.unsqueeze(-2)

        return y


