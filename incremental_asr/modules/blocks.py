from turtle import forward
import torch


class Conv1dBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: str,
        activation=torch.nn.LeakyReLU,
        dropout=0.2,
    ) -> None:
        super().__init__()

        self.conv = torch.nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.activation = activation()
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_norm = torch.nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        attention_heads: int,
        dropout: float = 0.2,
        activation=torch.nn.LeakyReLU
    ) -> None:
        super().__init__()
        self.sab = torch.nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = torch.nn.LayerNorm(embedding_dim)
        self.activation = activation()

    def forward(self, x):
        x = self.sab(x, x, x)[0]
        x = self.activation(x)
        x = self.norm(x)

        return x
