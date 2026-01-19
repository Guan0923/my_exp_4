import torch.nn as nn
import torch.nn.functional as F
import torch


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=3,
            padding=2,
            padding_mode="circular",
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class mHC_EncoderLayer_2(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        rate,
        iter,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(mHC_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(normalized_shape=(d_model, rate))
        self.norm2 = nn.LayerNorm(normalized_shape=(d_model, rate))
        self.dropout = nn.Dropout(dropout)

        self.activation = F.relu if activation == "relu" else F.gelu

        self.rate = rate
        self.iter = iter

        self.B = nn.Parameter(torch.ones((rate,))/rate)
        self.Am = nn.Parameter(torch.ones((rate,))/rate)
        self.Ar = nn.Parameter(torch.eye(rate))

        self.norm3 = nn.LayerNorm(normalized_shape=(d_model, rate))

    # X.shape (B, C, D, N)
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        A_r = self.Sinkhorn_Knopp(self.Ar, iter=self.iter)

        x = self.norm1(x)  # (BxCxDxN)

        x_in = torch.einsum("bcdn,n->bcd", x, self.Am)
        new_x, attn = self.attention(x_in, x_in, x_in, attn_mask=attn_mask, tau=tau, delta=delta)

        x = torch.einsum("bcdn,nm->bcdm", self.dropout(x), A_r) + torch.einsum(
            "n,bld->bldn", self.B, new_x
        )

        y = x = self.norm2(x)   
        y = torch.einsum("bcdn,n->bcd", y, self.Am)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        out = torch.einsum("bcdn,nm->bcdm", x, A_r) + torch.einsum(
            "n,bcd->bcdn", self.B, y
        )

        return self.norm3(out), attn

    def Sinkhorn_Knopp(self, A, iter=20, epsilon=1e-8):
        # A 的形状: [N, N]

        # 1. 数值稳定化与指数映射
        A = A - A.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        A = torch.exp(A)

        # 2. 迭代归一化 (关键：指定 dim=-1 和 dim=-2)
        for _ in range(iter):
            A = A / (A.sum(dim=-2, keepdim=True) + epsilon)
            A = A / (A.sum(dim=-1, keepdim=True) + epsilon)

        return A


class mHC_EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        rate,
        iter,
        layer_id,
        alpha_default=0.1,
        beta_default=0.1,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(mHC_EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model * rate)
        self.norm2 = nn.LayerNorm(d_model * rate)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

        self.rate = rate
        self.layer_id = layer_id
        self.iter = iter

        self.B = torch.ones((rate,))/rate

        self.Am = torch.zeros((rate,))
        self.Am[layer_id % rate] = 1.0
        self.Ar = torch.eye(rate)

        self.alpha_pre = nn.Parameter(torch.ones(1) * alpha_default)
        self.alpha_post = nn.Parameter(torch.ones(1) * alpha_default)
        self.alpha_res = nn.Parameter(torch.ones(1) * alpha_default)

        self.beta_pre = nn.Parameter(torch.ones(1) * beta_default)
        self.beta_post = nn.Parameter(torch.ones(1) * beta_default)
        self.beta_res = nn.Parameter(torch.ones(1) * beta_default)

        self.varphi_pre = nn.Parameter(
            torch.zeros(size=(d_model * rate, rate))
        )
        self.varphi_post = nn.Parameter(
            torch.zeros(size=(d_model * rate, rate))
        )
        self.varphi_res = nn.Parameter(
            torch.zeros(size=(d_model * rate, rate * rate))
        )

        self.norm3 = nn.LayerNorm(normalized_shape=(d_model, rate))

    # X.shape (B, C, D, N)
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        x_t = x.reshape(x.shape[0], x.shape[1], -1)
        x = self.norm1(x_t)  # (BxCx (DxN) )

        self.compute_Am_Ar_B(x)

        x = x.reshape(x.shape[0], x.shape[1], self.d_model, self.rate)
        x_in = torch.einsum("bcdn,n->bcd", x, self.Am)
        new_x, attn = self.attention(x_in, x_in, x_in, attn_mask=attn_mask, tau=tau, delta=delta)

        x = torch.einsum("bcdn,nm->bcdm", self.dropout(x), self.Ar) + torch.einsum(
            "n,bld->bldn", self.B, new_x
        )

        x = self.norm2(x.reshape(x.shape[0], x.shape[1], -1))   
        self.compute_Am_Ar_B(x)
        x = x.reshape(x.shape[0], x.shape[1], self.d_model, self.rate)
        x_in = torch.einsum("bcdn,n->bcd", x, self.Am)
        x_new = self.dropout(self.activation(self.conv1(x_in.transpose(-1, 1))))
        x_new = self.dropout(self.conv2(x_new).transpose(-1, 1))

        out = torch.einsum("bcdn,nm->bcdm", x, self.Ar) + torch.einsum(
            "n,bcd->bcdn", self.B, x_new
        )

        return self.norm3(out), attn

    # norm_h.shape (B, C, D * N)
    def compute_Am_Ar_B(self, norm_h):
        # pre-compute
        norm_h = norm_h.mean(dim=(0,1)) # norm_h.shape [D*N]
        Am_t = (self.alpha_pre * (norm_h @ self.varphi_pre) + self.beta_pre)
        self.Am = F.softmax(Am_t, dim=0)

        # post-compute
        B_t = (self.alpha_post * (norm_h @ self.varphi_post) + self.beta_post)
        # TODO self.B = F.softmax(B_t + self.B, dim=0) * 2 论文里
        self.B = F.softmax(B_t, dim=0)

        # res-compute
        A = (norm_h @ self.varphi_res).reshape(self.rate, self.rate)
        Ar_t = self.alpha_res * self.Sinkhorn_Knopp(A, iter=self.iter) + self.beta_res
        
        self.Ar = Ar_t

    def Sinkhorn_Knopp(self, A, iter=20, epsilon=1e-6):
        with torch.no_grad():
            # 1. 数值稳定化与指数映射
            A = A - A.max()
            A = torch.exp(A)

            # 2. 迭代归一化 (关键：指定 dim=-1 和 dim=-2)
            for _ in range(iter):
                A = A / (A.sum(dim=-2, keepdim=True) + epsilon)
                A = A / (A.sum(dim=-1, keepdim=True) + epsilon)

            return A

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn



class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    def __init__(
        self,
        self_attention,
        cross_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(
            self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        )
        x = self.norm1(x)

        x = x + self.dropout(
            self.cross_attention(
                x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta
            )[0]
        )

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(
                x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta
            )

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
