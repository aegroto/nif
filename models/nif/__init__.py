import torch

from torch import nn
from models.nif.utils import build_dumper
from modules.modulator import Modulator
from modules.modulator.identity import IdentityModulator
from modules.positional_encoder import PositionalEncoder

from modules.siren.utils import Sine, build_linear_initializer, initialize_first_layer

class NIF(nn.Module):
    def __init__(self, 
        input_features,
        hidden_sizes, 
        modulator_params,
        omega=30.0, 
        c=6.0,
        encoder_params=None,
        device=None,
        writer=None):
        super(NIF, self).__init__()

        self.device = device
        self.omega = omega
        self.c = c

        self.activation = Sine(1.0)

        self.encoder = PositionalEncoder(input_features, **encoder_params)

        self.head = nn.Linear(self.encoder.out_dim, hidden_sizes[0])

        body_layers = list()
        for i in range(0, len(hidden_sizes)-1):
            body_layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))

        self.body = nn.Sequential(*body_layers)

        if "disabled" in modulator_params and modulator_params["disabled"]:
            self.modulator = IdentityModulator()
        else:
            self.modulator = Modulator(input_features, writer=writer, **modulator_params)

        self.tail = nn.Linear(hidden_sizes[-1], 3)

        self.__dumper = build_dumper(writer, "period_map")

    def initialize_weights(self):
        with torch.no_grad():
            self.head.apply(initialize_first_layer)

            initializer = build_linear_initializer(self.c, self.omega)
            for body_layer in self.body:
                body_layer.apply(initializer)

            self.modulator.initialize_weights()

            self.tail.apply(initializer)

    def reshape_genesis(self, y):
        y = y.movedim(-1, 0)
        return y

    def mod_activation(self, period, y):
        y = self.modulator.group_features(y)
        y = self.activation(period * y)
        y = self.modulator.ungroup_features(y)
        return y

    def forward(self, x):
        m = self.modulator(x)
        period = m.add(self.omega)

        if self.__dumper and period is torch.Tensor:
            dump_period = period.movedim(-1, 0).squeeze(-1)
            self.__dumper(dump_period)
            self.__dumper.dump_scalar("abs_mean", period.abs().mean())

        y = self.encoder(x)
        y = self.head(y)
        y = self.mod_activation(period, y)

        for i in range(0, len(self.body)):
            y = self.body[i](y)
            y = self.mod_activation(period, y)

        y = self.tail(y)
        y = self.reshape_genesis(y)

        return y
