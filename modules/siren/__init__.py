from torch import nn

from modules.siren.utils import Sine, build_linear_initializer, initialize_first_layer

class Siren(nn.Module):
    def __init__(self, 
        omega,
        in_size, 
        out_size, 
        hidden_size,
        hidden_layers=0,
        final_activation=False
    ):
        super(Siren, self).__init__()

        head = SirenLayer(omega, in_size, hidden_size, initializer=initialize_first_layer)

        body = list()
        for _ in range(hidden_layers):
            body.append(SirenLayer(omega, hidden_size, hidden_size))

        tail = SirenLayer(omega, hidden_size, out_size, activate=final_activation)

        self.network = nn.Sequential(
            head,
            *body,
            tail
        )


    def forward(self, x):
        return self.network(x)


class SirenLayer(nn.Module):
    def __init__(self,
        omega, 
        in_feat, out_feat, 
        activate=True, 
        initializer=None
    ):
        super(SirenLayer, self).__init__()

        self.omega = omega
        self.linear = nn.Linear(in_feat, out_feat)

        if not initializer:
            initializer = build_linear_initializer(6.0, omega)

        self.initializer = initializer

        if activate:
            self.activation = Sine(omega)
        else:
            self.activation = None

    def initialize_weights(self):
        self.linear.apply(self.initializer)

    def __repr__(self):
        return f"SirenLayer({self.omega:.2f}, {self.linear})"

    def forward(self, x):
        y = self.linear(x)
        if self.activation:
            y = self.activation(y)

        return y
