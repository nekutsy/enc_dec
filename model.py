import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, layer_sizes: list[int], name: str = "autoencoder"):
        super().__init__()
        self.name = name
        self.layer_sizes = layer_sizes
        self.bottleneck_idx = layer_sizes.index(min(layer_sizes))

        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.BatchNorm1d(layer_sizes[i+1]))
                layers.append(nn.SiLU())

        self.net = nn.Sequential(*layers)

        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=0.5)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, x, to_bottleneck=False, from_bottleneck=False, z=None):
        if to_bottleneck:
            for i, layer in enumerate(self.net):
                x = layer(x)
                if i == self.bottleneck_idx:
                    return x
            return x
        elif from_bottleneck:
            x = z
            for i, layer in enumerate(self.net[self.bottleneck_idx+1:], start=self.bottleneck_idx+1):
                x = layer(x)
            return x
        else:
            return self.net(x)

    def encode(self, x):
        return self.forward(x, to_bottleneck=True)

    def decode(self, z):
        return self.forward(None, from_bottleneck=True, z=z)