import timm
import torch
import torch.nn as nn
from timm.layers import Linear, SelectAdaptivePool2d
from timm.models.ghostnet import GhostNet


class GhostFeatureNet(nn.Module):
    def __init__(self, *, pretrained: bool = True):
        super().__init__()
        self.ghostnet: GhostNet = timm.create_model(
            model_name="ghostnetv2_160",
            pretrained=pretrained,
        )

    def forward(self, x):
        return self.ghostnet.forward_features(x)

    def forward_head(self, x):
        return self.ghostnet.forward_head(x)

    @property
    def num_features(self):
        return self.ghostnet.num_features

    @property
    def head_hidden_size(self):
        return self.ghostnet.head_hidden_size


class GhostImageNet(nn.Module):
    def __init__(self, num_classes: int, *, pretrained: bool = True):
        super().__init__()
        self.ghost_feature_net = GhostFeatureNet(pretrained=pretrained)

        self.global_pool = SelectAdaptivePool2d(pool_type="avg")
        self.conv_head = nn.Conv2d(self.prev_chs, self.out_chs, 1, 1, 0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten(1)
        self.classifier = Linear(self.out_chs, num_classes)

    def forward(self, x):
        x = self.forward_feature(x)
        x = self.forward_hidden(x)
        x = self.classifier(x)
        return x

    def forward_feature(self, x):
        x = self.ghost_feature_net(x)
        return x

    def forward_hidden(self, x):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act(x)
        x = self.flatten(x)
        return x

    @property
    def prev_chs(self):
        return self.ghost_feature_net.num_features

    @property
    def out_chs(self):
        return self.ghost_feature_net.head_hidden_size


class GhostVideoNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        self.ghost_image_net = GhostImageNet(num_classes, pretrained=pretrained)

        self.lstm = nn.LSTM(
            input_size=self.out_chs,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = Linear(lstm_output_size, num_classes)

    def forward(self, x):
        # Assume x is of shape (batch, sequence, channels, height, width)
        batch_size, sequence_length, c, h, w = x.shape
        x = x.view(batch_size * sequence_length, c, h, w)

        with torch.no_grad():
            features = self.forward_image_feature(
                x
            )  # Features of shape (batch_size * sequence_length, features, H', W')
        features = self.forward_image_hidden(features)

        # Global average pooling
        features = features.view(batch_size, sequence_length, -1)

        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out[:, -1, :]  # Get the last output of the LSTM
        out = self.fc(lstm_out)
        return out

    def forward_image_feature(self, x):
        x = self.ghost_image_net.forward_feature(x)
        return x

    def forward_image_hidden(self, x):
        x = self.ghost_image_net.forward_hidden(x)
        return x

    @property
    def prev_chs(self):
        return self.ghost_image_net.prev_chs

    @property
    def out_chs(self):
        return self.ghost_image_net.out_chs


__all__ = [
    "GhostImageNet",
    "GhostVideoNet",
]


# Example usage
if __name__ == "__main__":
    num_classes = 10

    # Test GhostImageNet
    image_model = GhostImageNet(num_classes)
    image_input = torch.randn(1, 3, 256, 256)
    image_output = image_model(image_input)
    print(f"Image model output shape: {image_output.shape}")

    # Test GhostVideoNet
    video_model = GhostVideoNet(
        num_classes=num_classes,
        hidden_size=128,
        num_layers=2,
        bidirectional=True,
    )
    video_input = torch.randn(2, 5, 3, 256, 256)  # 2 videos, 5 frames, 3x224x224
    video_output = video_model(video_input)
    print(f"Video model output shape: {video_output.shape}")
