import itertools
import math
from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F

class NConv(nn.Module):

    def forward(self, x, A):
        x = torch.matmul(x.transpose(2, 3), A).transpose(2, 3)
        return x.contiguous()

class GCN(nn.Module):

    def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
        super().__init__()
        self.nconv = NConv()
        self.order = order
        self.mlp = nn.Conv2d((order * support_len + 1) * c_in, c_out, kernel_size=(1, 1))
        self.dropout = dropout
        self.support_len = support_len

    def forward(self, x, supports):
        out = [x]
        for A in supports:
            x1 = self.nconv(x, A)
            out.append(x1)
            for _ in range(2, self.order + 1):
                x1 = self.nconv(x1, A)
                out.append(x1)
        h = torch.cat(out, dim=1)
        h = self.mlp(h)
        h = F.softplus(h)
        h = F.dropout(h, self.dropout, self.training)
        return h

class GWTCNBlock(nn.Module):

    def __init__(self, residual_channels, dilation_channels, kernel_size, dilation, gcn_bool, supports_len, dropout):
        super().__init__()
        self.gcn_bool = gcn_bool
        pad = (kernel_size - 1) * dilation
        self.filter_conv = nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=(1, dilation), padding=(0, pad))
        self.gate_conv = nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=(1, dilation), padding=(0, pad))
        self.residual_conv = nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(residual_channels)
        self.gconv = GCN(dilation_channels, residual_channels, dropout, support_len=supports_len) if gcn_bool else None

    def forward(self, x, supports=None):
        residual = x
        f = torch.tanh(self.filter_conv(residual))
        g = torch.sigmoid(self.gate_conv(residual))
        x = f * g
        x = x[..., :residual.size(-1)]
        if self.gcn_bool and supports is not None and (len(supports) > 0):
            x = self.gconv(x, supports)
        else:
            x = self.residual_conv(x)
        x = x + residual
        x = self.bn(x)
        return x

class MiniGWTCN(nn.Module):

    def __init__(self, feats, kernel_size=3, dilations=(1, 2), residual_channels=32, dilation_channels=32, gcn_bool=True, addaptadj=True, supports=None, apt_rank=10, dropout=0.2, device='cpu'):
        super().__init__()
        self.feats = feats
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.device = device
        self.dropout = dropout
        self.start = nn.Conv2d(1, residual_channels, kernel_size=(1, 1))
        self.blocks = nn.ModuleList()
        self.supports = supports or []
        self.supports_len = len(self.supports)
        if gcn_bool and addaptadj:
            self.nodevec1 = nn.Parameter(torch.randn(feats, apt_rank, device=device))
            self.nodevec2 = nn.Parameter(torch.randn(apt_rank, feats, device=device))
            self.supports_len += 1
        for d in dilations:
            self.blocks.append(GWTCNBlock(residual_channels, dilation_channels, kernel_size, d, gcn_bool=self.gcn_bool, supports_len=self.supports_len, dropout=dropout))
        self.end = nn.Conv2d(residual_channels, 1, kernel_size=(1, 1))

    def _supports(self):
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(F.relu(self.nodevec1 @ self.nodevec2), dim=1)
            return self.supports + [adp]
        return self.supports

    def forward(self, x):
        x4 = x.unsqueeze(1)
        x4 = self.start(x4)
        supports = self._supports()
        for blk in self.blocks:
            x4 = blk(x4, supports)
        x4 = self.end(x4)
        out = x4[:, 0, :, :]
        return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model).float() * (-math.log(10000.0) / d_model))
        pe += torch.sin(position * div_term)
        pe += torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, pos=0):
        x = x + self.pe[pos:pos + x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=16, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.att = None
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.LeakyReLU(True)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal: bool=False, **kwargs):
        (src2, att) = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=True, is_causal=is_causal)
        self.att = att
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        return src

def _safe_extract_attn(encoder: nn.Module) -> Optional[torch.Tensor]:
    if encoder is None or not hasattr(encoder, 'layers'):
        return None
    try:
        layer0 = encoder.layers[0]
        if hasattr(layer0, 'att') and layer0.att is not None:
            return layer0.att
    except Exception:
        return None
    return None

def _build_unordered_rule_masks(n_feats: int, max_rules: int=16) -> torch.Tensor:
    n_feats = int(max(1, n_feats))
    max_rules = int(max(1, max_rules))
    masks = []
    feat_ids = list(range(n_feats))
    for k in range(1, n_feats + 1):
        for combo in itertools.combinations(feat_ids, k):
            mask = torch.zeros(n_feats, dtype=torch.float32)
            mask[list(combo)] = 1.0
            masks.append(mask)
            if len(masks) >= max_rules:
                return torch.stack(masks, dim=0)
    return torch.stack(masks, dim=0)

class AttentionGuidedTSKRiskDecoder(nn.Module):

    def __init__(self, n_feats: int, n_heads: int=3, n_rules: int=16, rule_tau: float=0.7, rule_masks: Optional[torch.Tensor]=None, mask_soft: float=0.1, recon_risk_ratio: float=0.7):
        super().__init__()
        self.n_feats = int(n_feats)
        self.n_heads = int(n_heads)
        self.n_inputs = 2 * self.n_feats
        self.rule_tau = float(max(0.001, rule_tau))
        self.mask_soft = float(min(1.0, max(0.0, mask_soft)))
        self.recon_risk_ratio = float(min(1.0, max(0.0, recon_risk_ratio)))
        if rule_masks is not None:
            masks = torch.as_tensor(rule_masks, dtype=torch.float32)
            if masks.dim() != 2 or masks.shape[1] != self.n_feats:
                raise ValueError(f'rule_masks must be (R, F), got {tuple(masks.shape)}')
            self.n_rules = int(max(1, masks.shape[0]))
            self.register_buffer('rule_masks', masks)
        else:
            self.n_rules = int(max(2, n_rules))
            self.register_buffer('rule_masks', torch.empty(0))
        self.rule_centers = nn.Parameter(torch.randn(self.n_heads, self.n_rules, self.n_inputs) * 0.1)
        self.rule_log_sigmas = nn.Parameter(torch.zeros(self.n_heads, self.n_rules, self.n_inputs))
        self.rule_consequents = nn.Parameter(torch.randn(self.n_heads, self.n_rules, self.n_inputs + 1) * 0.05)
        self.alpha_to_rule = nn.Linear(self.n_feats, self.n_heads * self.n_rules)
        self.context_to_reconstruction = nn.Linear(self.n_inputs, self.n_feats)
        self.to_reconstruction = nn.Sequential(nn.Linear(self.n_heads, self.n_feats), nn.Sigmoid())
        self.risk_log_temp = nn.Parameter(torch.zeros(1))

    def _feature_contribution(self, z: torch.Tensor, attn_weights: Optional[torch.Tensor]) -> torch.Tensor:
        z_bwf = z.permute(1, 0, 2)
        (bsz, win, _) = z_bwf.shape
        if attn_weights is None:
            time_importance = torch.full((bsz, win), 1.0 / float(max(1, win)), device=z.device, dtype=z.dtype)
        else:
            if attn_weights.dim() == 2:
                attn_bww = attn_weights.unsqueeze(0).expand(bsz, -1, -1)
            elif attn_weights.dim() == 3:
                attn_bww = attn_weights
            else:
                attn_bww = None
            if attn_bww is None:
                time_importance = torch.full((bsz, win), 1.0 / float(max(1, win)), device=z.device, dtype=z.dtype)
            else:
                time_importance = attn_bww.mean(dim=1)
                time_importance = time_importance / (time_importance.sum(dim=1, keepdim=True) + 1e-12)
        return (z_bwf.abs() * time_importance.unsqueeze(-1)).sum(dim=1)

    def _build_fuzzy_input(self, z: torch.Tensor) -> torch.Tensor:
        z_bwf = z.permute(1, 0, 2)
        z_mean = z_bwf.mean(dim=1)
        z_std = z_bwf.std(dim=1).clamp(min=1e-06)
        return torch.cat([z_mean, z_std], dim=1)

    def forward(self, z: torch.Tensor, attn_weights: Optional[torch.Tensor]=None):
        alpha = self._feature_contribution(z, attn_weights)
        x_fuzzy = self._build_fuzzy_input(z)
        bsz = x_fuzzy.shape[0]
        x_exp = x_fuzzy.unsqueeze(1).unsqueeze(1)
        centers = self.rule_centers.unsqueeze(0)
        sigmas = torch.exp(self.rule_log_sigmas).clamp(min=0.0001).unsqueeze(0)
        if self.rule_masks.numel() > 0:
            mask = self.rule_masks.unsqueeze(0).unsqueeze(0)
            mask_inputs = torch.cat([mask, mask], dim=-1)
            soft = self.mask_soft
            soft_mask_inputs = mask_inputs * (1.0 - soft) + soft
            x_exp = x_exp * soft_mask_inputs
            centers = centers * soft_mask_inputs
        sq_dist = ((x_exp - centers) / sigmas) ** 2
        log_firing = -0.5 * sq_dist.sum(dim=-1)
        firing = torch.exp(log_firing.clamp(min=-30.0, max=10.0))
        rule_logits = self.alpha_to_rule(alpha).view(bsz, self.n_heads, self.n_rules)
        rule_prior = torch.softmax(rule_logits / self.rule_tau, dim=-1)
        firing = firing * (rule_prior + 1e-06)
        norm_firing = firing / (firing.sum(dim=-1, keepdim=True) + 1e-12)
        x_bias = torch.cat([torch.ones(bsz, 1, device=x_fuzzy.device, dtype=x_fuzzy.dtype), x_fuzzy], dim=1)
        if self.rule_masks.numel() > 0:
            soft = self.mask_soft
            feat_mask = self.rule_masks.to(device=x_fuzzy.device, dtype=x_fuzzy.dtype)
            feat_mask = feat_mask * (1.0 - soft) + soft
            mask_bias = torch.cat([torch.ones(self.n_rules, 1, device=x_fuzzy.device, dtype=x_fuzzy.dtype), feat_mask.repeat(1, 2)], dim=1)
        else:
            mask_bias = None
        consequent = self.rule_consequents.unsqueeze(0)
        if mask_bias is not None:
            rule_outputs = torch.sum(consequent * x_bias.unsqueeze(1).unsqueeze(1) * mask_bias.unsqueeze(0).unsqueeze(0), dim=-1)
        else:
            rule_outputs = torch.sum(consequent * x_bias.unsqueeze(1).unsqueeze(1), dim=-1)
        raw_scores = torch.sum(norm_firing * rule_outputs, dim=-1)
        temp = torch.exp(self.risk_log_temp).clamp(min=0.5, max=2.5)
        risk_membership = F.softmax(raw_scores / temp, dim=1)
        rec_from_risk = self.to_reconstruction(risk_membership)
        rec_from_context = torch.sigmoid(self.context_to_reconstruction(x_fuzzy))
        r = self.recon_risk_ratio
        x_mapped = (r * rec_from_risk + (1.0 - r) * rec_from_context).unsqueeze(1)
        return (x_mapped, risk_membership, norm_firing)

class TSK_SPCN(nn.Module):

    def __init__(self, feats, n_window=10, lr=0.002, device='cpu', supports=None, recon_risk_ratio=0.7, n_rules=None, rule_tau=0.7, gwtcn_blocks=None, mask_soft=0.1, transformer_ff_dim=None, transformer_heads=None):
        super().__init__()
        self.name = 'TSK-SPCN'
        self.lr = lr
        self.batch = 128
        self.n_feats = feats
        self.n_window = n_window
        self.recon_risk_ratio = float(recon_risk_ratio)
        self.n_rules = None if n_rules is None else max(1, int(n_rules))
        self.rule_tau = float(max(0.001, rule_tau))
        self.gwtcn_blocks = None if gwtcn_blocks is None else max(1, int(gwtcn_blocks))
        self.mask_soft = float(min(1.0, max(0.0, mask_soft)))
        self.transformer_ff_dim = 64 if transformer_ff_dim is None else int(transformer_ff_dim)
        self.transformer_heads = feats if transformer_heads is None else int(transformer_heads)
        if self.transformer_ff_dim < 1:
            raise ValueError('transformer_ff_dim must be positive')
        if self.transformer_heads < 1 or feats % self.transformer_heads:
            raise ValueError('transformer_heads must be positive and divide sensor count')

        def _dilations(base):
            depth = self.gwtcn_blocks or len(base)
            values = list(base)
            while len(values) < depth:
                values.append(values[-1] * 2)
            return tuple(values[:depth])
        self.l_tcn = MiniGWTCN(feats, kernel_size=4, dilations=_dilations((1, 2)), supports=supports, device=device, dropout=0.2)
        self.g_tcn = MiniGWTCN(feats, kernel_size=3, dilations=_dilations((2, 4)), supports=supports, device=device, dropout=0.2)
        self.pos_encoder = PositionalEncoding(feats, 0.1, self.n_window)
        nhead = self.transformer_heads
        enc1 = TransformerEncoderLayer(d_model=feats, nhead=nhead, dim_feedforward=self.transformer_ff_dim, dropout=0.1)
        enc2 = TransformerEncoderLayer(d_model=feats, nhead=nhead, dim_feedforward=self.transformer_ff_dim, dropout=0.1)
        self.transformer_encoder1 = nn.TransformerEncoder(enc1, num_layers=1)
        self.transformer_encoder2 = nn.TransformerEncoder(enc2, num_layers=1)
        self.fcn = nn.Linear(feats, feats)
        rule_budget = self.n_rules if self.n_rules is not None else max(16, feats)
        rule_masks = _build_unordered_rule_masks(feats, max_rules=rule_budget)
        self.anfis_decoder1 = AttentionGuidedTSKRiskDecoder(n_feats=feats, n_heads=3, rule_tau=self.rule_tau, rule_masks=rule_masks, mask_soft=self.mask_soft, recon_risk_ratio=self.recon_risk_ratio)
        self.anfis_decoder2 = AttentionGuidedTSKRiskDecoder(n_feats=feats, n_heads=3, rule_tau=self.rule_tau, rule_masks=rule_masks, mask_soft=self.mask_soft, recon_risk_ratio=self.recon_risk_ratio)

    def callback(self, src, x1_mapped):
        src2 = src + x1_mapped.permute(0, 2, 1)
        g_atts = self.g_tcn(src2)
        src2 = g_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src2 = self.pos_encoder(src2)
        memory = self.transformer_encoder2(src2)
        return memory

    def forward(self, src):
        l_atts = self.l_tcn(src)
        src1 = l_atts.permute(2, 0, 1) * math.sqrt(self.n_feats)
        src1 = self.pos_encoder(src1)
        z1 = self.transformer_encoder1(src1)
        c1 = z1 + self.fcn(z1)
        attn1 = _safe_extract_attn(self.transformer_encoder1)
        (x1_mapped, risk_score1, _) = self.anfis_decoder1(c1, attn_weights=attn1)
        z2 = self.fcn(self.callback(src, x1_mapped))
        c2 = z2 + self.fcn(z2)
        attn2 = _safe_extract_attn(self.transformer_encoder2)
        (x2_mapped, risk_score2, strengths2) = self.anfis_decoder2(c2, attn_weights=attn2)
        return (x1_mapped, x2_mapped, risk_score1, risk_score2, strengths2)
