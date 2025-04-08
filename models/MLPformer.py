import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.RevIN import RevIN
from layers.MLP_encoder import MLP_encoder
from layers.MLP_decoder import MLP_decoder
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp, series_decomp_multi

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_weights(x_enc):
    harmony_num = 3
    freq = torch.fft.rfft(x_enc - torch.mean(x_enc, dim=1, keepdim=True), dim=1)
    freq = torch.abs(freq)  # 得到信号在各个频率上的能量分布
    _freq = freq.clone()
    _freq[:, :3, :] = 0
    _freq[:, freq.shape[1] // harmony_num:, :] = 0  # 通过将 _freq 张量中部分频率分量设为 0，来过滤掉特定的频率成分，以实现信号的优化或改进。

    max_amp, indices = torch.max(_freq, dim=1, keepdim=True)
    amp_sum = torch.zeros_like(max_amp).to(x_enc.device)

    for i in range(harmony_num):
        har = (i + 1) * indices  # 获取当前谐波的索引
        har_value = torch.gather(freq, 1, har) ** 2  # 获取谐波值
        amp_sum = amp_sum + har_value

    total_sum = torch.sum(freq ** 2, dim=1, keepdim=True) + 1e-5  # 计算 freq 在特征维度上的平方和
    weights = amp_sum / total_sum  # 通过归一化幅度和来计算权重，使其在后续处理或使用时更为合理和稳定。

    return weights


class Model(nn.Module):
    """
    FEDformer performs the attention mechanism on frequency domain and achieved O(N) complexity
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.version = configs.version
        self.mode_select = configs.mode_select
        self.modes = configs.modes
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.d_model = configs.d_model
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)
        #self.t_model = T_Block(configs)
        #self.f_model = F_Block(configs)
        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.residual_connection = configs.residual_connection
        self.device = torch.device('cuda:{}'.format(configs.gpu))
        self.k = configs.k
        self.num_nodes = configs.num_nodes
        self.d_ff = configs.d_ff

        # Decomp
        kernel_size = configs.moving_avg
        if isinstance(kernel_size, list):
            self.decomp = series_decomp_multi(kernel_size)
        else:
            self.decomp = series_decomp(kernel_size)

        # Embedding
        # The series-wise connection inherently contains the sequential information.
        # Thus, we can discard the position embedding of transformers.
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)


        encoder_self_att = MLP_encoder(configs)
        decoder_self_att = MLP_decoder(configs)
        decoder_cross_att = AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                               output_attention=False)



        # Encoder
        enc_modes = int(min(configs.modes, configs.seq_len//2))
        dec_modes = int(min(configs.modes, (configs.seq_len//2+configs.pred_len)//2))
        print('enc_modes: {}, dec_modes: {}'.format(enc_modes, dec_modes))

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        encoder_self_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        decoder_self_att,
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        decoder_cross_att,
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        #if self.revin:
           # x_enc = self.revin_layer(x_enc, 'norm')
        #x_enc = self.start_fc(x.unsqueeze(-1))

        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).to(device)  # cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = F.pad(seasonal_init[:, -self.label_len:, :], (0, 0, 0, self.pred_len))
        # enc
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final

        #weights = calculate_weights(x_dec)
        #print("weights:",weights)
        #dec_out = seasonal_part * weights + trend_part * (0.5 - weights)
        dec_out = trend_part + seasonal_part

        #if self.revin:
            #dec_out = self.revin_layer(dec_out, 'denorm')

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


if __name__ == '__main__':
    class Configs(object):
        ab = 0
        modes = 32
        mode_select = 'random'
        # version = 'Fourier'
        version = 'Wavelets'
        moving_avg = [12, 24]
        L = 1
        base = 'legendre'
        cross_activation = 'tanh'
        seq_len = 96
        label_len = 48
        pred_len = 96
        output_attention = True
        enc_in = 7
        dec_in = 7
        d_model = 16
        embed = 'timeF'
        dropout = 0.05
        freq = 'h'
        factor = 1
        n_heads = 8
        d_ff = 16
        e_layers = 3
        d_layers = 3
        c_out = 7
        activation = 'gelu'
        wavelet = 0

    configs = Configs()
    model = Model(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    enc = torch.randn([3, configs.seq_len, 7])
    enc_mark = torch.randn([3, configs.seq_len, 4])

    dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7])
    dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4])
    out = model.forward(enc, enc_mark, dec, dec_mark)
    print(out)


