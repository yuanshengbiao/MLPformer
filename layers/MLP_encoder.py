import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN_MLP import RevIN


class MLP_encoder(nn.Module):

    def __init__(self, configs):
        super(MLP_encoder, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.embed_size = configs.embed_size
        self.batch_size = configs.batch_size
        self.n_heads = configs.n_heads
        self.d_model = configs.d_model

        self.hidden_size = configs.hidden_size
        self.dropout = configs.dropout
        self.band_width = 96
        self.scale = 0.02
        self.sparsity_threshold = 0.01

        self.revin_layer = RevIN(configs.d_model, affine=True, subtract_last=False)
        self.embedding = nn.Linear(self.seq_len, self.embed_size)
        self.token = nn.Conv1d(in_channels=self.seq_len, out_channels=self.embed_size, kernel_size=(1,))

        self.w = nn.Parameter(self.scale * torch.randn(2, self.embed_size))
        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.embed_size))

        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.embed_size)
        )

        #self.output = nn.Linear(self.embed_size, self.pred_len)
        self.output = nn.Linear(self.embed_size, self.seq_len)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.layernorm1 = nn.LayerNorm(self.embed_size)
        self.dropout = nn.Dropout(self.dropout)

    def tokenEmbed(self, x):
        x = self.token(x)
        return x

    def MLP(self, x):
        B, N, _ = x.shape
        o1_real = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device=x.device)
        o1_imag = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device=x.device)

        o2_real = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device=x.device)
        o2_imag = torch.zeros([B, N // 2 + 1, self.embed_size],
                              device=x.device)

        o1_real = F.relu(
            torch.einsum('bid,d->bid', x.real, self.w[0]) - \
            torch.einsum('bid,d->bid', x.imag, self.w[1]) + \
            self.rb1
        )

        o1_imag = F.relu(
            torch.einsum('bid,d->bid', x.imag, self.w[0]) + \
            torch.einsum('bid,d->bid', x.real, self.w[1]) + \
            self.ib1
        )

        o2_real = (
                torch.einsum('bid,d->bid', o1_real, self.w1[0]) - \
                torch.einsum('bid,d->bid', o1_imag, self.w1[1]) + \
                self.rb2
        )

        o2_imag = (
                torch.einsum('bid,d->bid', o1_imag, self.w1[0]) + \
                torch.einsum('bid,d->bid', o1_real, self.w1[1]) + \
                self.ib2
        )

        o3_real = (
                torch.einsum('bid,d->bid', o2_real, self.w1[0]) - \
                torch.einsum('bid,d->bid', o2_imag, self.w1[1]) + \
                self.rb2
        )

        o3_imag = (
                torch.einsum('bid,d->bid', o2_imag, self.w1[0]) + \
                torch.einsum('bid,d->bid', o2_real, self.w1[1]) + \
                self.ib2
        )

        y = torch.stack([o3_real, o3_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [Batch, Input length, Channel]
        #print("x:", x.shape)
        x = x.view(x.size(0), x.size(1), -1)
        B, L, N = x.shape
        z = x
        z = self.revin_layer(z, 'norm')
        x = z

        x = x.permute(0, 2, 1)
        #print("encoder_x.shape:",x.shape)
        x = self.embedding(x)  # B, N, D
        x = self.layernorm(x)
        x = torch.fft.rfft(x, dim=1, norm='ortho')
        weight = self.MLP(x)
        x = x * weight + weight
        x = torch.fft.irfft(x, n=N, dim=1, norm="ortho")
        x = self.layernorm1(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.output(x)
        x = x.permute(0, 2, 1)

        z = x
        z = self.revin_layer(z, 'denorm')
        x = z
        #print("encoder.shape:",x.shape)
        x = x.reshape(self.batch_size, self.n_heads, int(self.d_model / self.n_heads), self.seq_len)
        #print("TextFilter output:",x.shape)

        return (x,None)
