import torch
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )

        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(
            self.weight,
            mean=0,
            std=std,
            a= -3 * std,
            b=3 * std,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T

class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None,):
        super().__init__()

        # Add the embedding table

        # This creates one row for every token, and each row has embedding_dim numbers
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )

        # Fills the empty table with reasonable random starting numbers,
        # mostly near zero, and prevents values between -3 or above 3
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        mean_square = x.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(mean_square + self.eps)
        result = (x / rms) * self.weight
        return result.to(in_dtype)

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, device=None, dtype=None):
        super().__init__()
        if d_ff is None:
            d_ff = int(8 * d_model / 3)
            d_ff = ((d_ff + 63) // 64) * 64
        self.d_ff = d_ff

        # for the gate
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # for the feature
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        # for project back to original size
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        gate = gate * torch.sigmoid(gate)

        features = self.w3(x)

        hidden = gate * features
        output = self.w2(hidden)

        return output

class RotaryPositionalEmbedding(nn.Module):
    # theta: big theta in the formula
    # d_k: number of dimension of one query / key vector, vector size
    # max_seq_len: number of positions we support, position: token index
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        # Starting dimension index
        dims = torch.arange(0, d_k, 2, device=device)
        # If d_k = 6, dims = [0, 2, 4]
        # Rotation speed for every pair
        freqs = 1.0 / (theta ** (dims / d_k))

        positions = torch.arange(max_seq_len, device=device)
        # Token indices, if max_swq_len = 5,
        # positions = [0, 1, 2, 3, 4]

        angles = positions[:, None] * freqs[None, :]
        # We get a 2d matrix, row is the token index, column is the dimension index
        # angles store the small theta i, k

        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.cos: torch.Tensor
        self.sin: torch.Tensor

        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]


        # This is a multi dimension matrix.
        # We get the even and odd indices element from last dimension to extract the data
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        # Put matching even / odd values back together as pairs
        # Like rotated_even: [a, c, e]
        # rotated odd: [b, d, f]
        # We get:
        # [
        #    [a, b],
        #    [c, d],
        #    [e, f],
        # ]
        rotated = torch.stack([rotated_even, rotated_odd,], dim=-1)

        # [a, b, c, d, e, f]
        rotated = rotated.flatten(-2)
        return rotated
