import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.masking import TriangularCausalMask, ProbMask
from models.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from models.decoder import Decoder, DecoderLayer
from models.attn import FullAttention, ProbAttention, AttentionLayer
from models.embed import DataEmbedding






#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■

class Informer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=3, d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        
        self.pred_len = out_len                                    #self.pred_len = 10
        self.attn = attn                                           #self.attn = prob
        self.output_attention = output_attention                   #self.output_attention = False

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        
        enc_out = self.enc_embedding(x_enc, x_mark_enc)                                              # x_enc:(32,96,8) ／ x_mark_enc:(32,96,5)
                                                                                                     # → enc_out:(32,96,512)
            
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)                              # enc_out:(32,96,512) ／ attn_mask:None
                                                                                                     # → enc_out:(32,48,512) ／ attn_mask:[None, None]

        dec_out = self.dec_embedding(x_dec, x_mark_dec)                                              # x_dec:(32,58,8) ／ x_mark_dec:(32,58,5)
                                                                                                     # → dec_out: torch.size(58,512)の32のリスト
        
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)      # dec_self_mask:None ／ dec_enc_mask:None
                                                                                                     # → dec_out:(32,58,512)
        
        dec_out = self.projection(dec_out)                                                           # dec_out:(32,58,512)
                                                                                                     # → dec_out:(32,58,8)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:                                                                    # self.output_attention = False のため不実行
            return dec_out[:,-self.pred_len:,:], attns                                               # ↓
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]                                          # return:(32,10,8) … dec_out:(32,58,8)の最後10要素
        
#//////////////////////////////////////////////////////////////////////////////////////////////////////
# ▼self.enc_embedding
# DataEmbedding
#   ┣━(value_embedding)   : TokenEmbedding((tokenConv): Conv1d(8, 512, kernel_size=(3,), stride=(1,), padding=(1,), padding_mode=circular))
#   ┣━(position_embedding): PositionalEmbedding()
#   ┣━(temporal_embedding): TimeFeatureEmbedding((embed): Linear(in_features=5, out_features=512, bias=True))
#   ┗━(dropout)           : Dropout(p=0.05, inplace=False)

# ▼self.dec_embedding
# DataEmbedding
#   ┣━(value_embedding)   : TokenEmbedding((tokenConv): Conv1d(8, 512, kernel_size=(3,), stride=(1,), padding=(1,), padding_mode=circular))
#   ┣━(position_embedding): PositionalEmbedding()
#   ┣━(temporal_embedding): TimeFeatureEmbedding((embed): Linear(in_features=5, out_features=512, bias=True))
#   ┗━(dropout)           : Dropout(p=0.05, inplace=False)

# ▼self.encoder
# Encoder
#  (attn_layers): ModuleList
#   ┣━(0): EncoderLayer
#   ┃  ┣━(attention): AttentionLayer
#   ┃  ┃  ┣━(inner_attention): ProbAttention((dropout): Dropout(p=0.05, inplace=False))
#   ┃  ┃  ┣━(query_projection): Linear(in_features=512, out_features=512, bias=True)
#   ┃  ┃  ┣━(key_projection): Linear(in_features=512, out_features=512, bias=True)
#   ┃  ┃  ┣━(value_projection): Linear(in_features=512, out_features=512, bias=True)
#   ┃  ┃  ┗━(out_projection): Linear(in_features=512, out_features=512, bias=True)
#   ┃  ┃    
#   ┃  ┣━(conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
#   ┃  ┣━(conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
#   ┃  ┣━(norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#   ┃  ┣━(norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#   ┃  ┗━(dropout): Dropout(p=0.05, inplace=False)
#   ┃ 
#   ┗━(1): EncoderLayer
#      ┣━(attention): AttentionLayer
#      ┃  ┣━(inner_attention): ProbAttention((dropout): Dropout(p=0.05, inplace=False))
#      ┃  ┣━(query_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃  ┣━(key_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃  ┣━(value_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃  ┗━(out_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃ 
#      ┣━(conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
#      ┣━(conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
#      ┣━(norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#      ┣━(norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#      ┗━(dropout): Dropout(p=0.05, inplace=False)
#      
#  (conv_layers): ModuleList
#   ┗━(0): ConvLayer
#      ┣━(downConv): Conv1d(512, 512, kernel_size=(3,), stride=(1,), padding=(1,), padding_mode=circular)
#      ┣━(norm): BatchNorm1d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
#      ┣━(activation): ELU(alpha=1.0)
#      ┗━(maxPool): MaxPool1d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
#
#  (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)


# ▼self.decoder
# Decoder
#  (layers): ModuleList(
#   ┗━(0): DecoderLayer
#      ┣━(self_attention): AttentionLayer
#      ┃  ┣━(inner_attention): ProbAttention((dropout): Dropout(p=0.05, inplace=False))
#      ┃  ┣━(query_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃  ┣━(key_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃  ┣━(value_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃  ┗━(out_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃
#      ┣━(cross_attention): AttentionLayer(
#      ┃  ┣━(inner_attention): FullAttention((dropout): Dropout(p=0.05, inplace=False))
#      ┃  ┣━(query_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃  ┣━(key_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃  ┣━(value_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃  ┗━(out_projection): Linear(in_features=512, out_features=512, bias=True)
#      ┃
#      ┣━(conv1): Conv1d(512, 2048, kernel_size=(1,), stride=(1,))
#      ┣━(conv2): Conv1d(2048, 512, kernel_size=(1,), stride=(1,))
#      ┣━(norm1): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#      ┣━(norm2): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#      ┣━(norm3): LayerNorm((512,), eps=1e-05, elementwise_affine=True)
#      ┗━(dropout): Dropout(p=0.05, inplace=False)
#
#  (norm): LayerNorm((512,), eps=1e-05, elementwise_affine=True)

# ▼self.projection
#   Linear(in_features=512, out_features=8, bias=True)
#//////////////////////////////////////////////////////////////////////////////////////////////////////

#■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

class InformerStack(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, out_len, 
                factor=5, d_model=512, n_heads=8, e_layers=[3,2,1], d_layers=2, d_ff=512, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='gelu',
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(InformerStack, self).__init__()
        self.pred_len = out_len
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        # Encoder

        inp_lens = list(range(len(e_layers))) # [0,1,2,...] you can customize here
        encoders = [
            Encoder(
                [
                    EncoderLayer(
                        AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                    d_model, n_heads, mix=False),
                        d_model,
                        d_ff,
                        dropout=dropout,
                        activation=activation
                    ) for l in range(el)
                ],
                [
                    ConvLayer(
                        d_model
                    ) for l in range(el-1)
                ] if distil else None,
                norm_layer=torch.nn.LayerNorm(d_model)
            ) for el in e_layers]
        self.encoder = EncoderStack(encoders, inp_lens)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(Attn(True, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=mix),
                    AttentionLayer(FullAttention(False, factor, attention_dropout=dropout, output_attention=False), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for l in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # self.end_conv1 = nn.Conv1d(in_channels=label_len+out_len, out_channels=out_len, kernel_size=1, bias=True)
        # self.end_conv2 = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=1, bias=True)
        self.projection = nn.Linear(d_model, c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)
        
        # dec_out = self.end_conv1(dec_out)
        # dec_out = self.end_conv2(dec_out.transpose(2,1)).transpose(1,2)
        if self.output_attention:
            return dec_out[:,-self.pred_len:,:], attns
        else:
            return dec_out[:,-self.pred_len:,:] # [B, L, D]
