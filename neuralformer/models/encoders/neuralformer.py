import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
from neuralformer.data_process.position_encoding import Embedder

import math
import algos

from torch import Tensor
from typing import Optional, List


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    dropout: Optional[nn.Module] = None,
    rel_pos_bias: Optional[Tensor] = None,
) -> Tensor:
    d_k = query.size(-1)
    # (b, n_head, l_q, d_per_head) * (b, n_head, d_per_head, l_k)
    attn = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if rel_pos_bias is not None:
        attn = attn * (1 + rel_pos_bias)
    attn = F.softmax(attn, dim=-1)
    if dropout is not None:
        attn = dropout(attn)  # (b, n_head, l_q, l_k)
    return torch.matmul(attn, value)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float,
        q_learnable: bool,
        rel_pos_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.d_k = dim // n_head  # default: 32

        self.linears = nn.ModuleList([nn.Linear(dim, dim), nn.Linear(dim, dim)])
        if q_learnable:
            self.linears.append(nn.Identity())
        else:
            self.linears.append(nn.Linear(dim, dim))

        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rel_pos_bias = rel_pos_bias
        if rel_pos_bias:
            self.rel_pos_encoder_forward = nn.Embedding(10, self.n_head, padding_idx=9)
            self.rel_pos_encoder_backward = nn.Embedding(10, self.n_head, padding_idx=9)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        rel_pos: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size = query.size(0)

        key, value, query = [
            l(x).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (key, value, query))
        ]
        if self.rel_pos_bias:
            rel_pos_bias = (
                self.rel_pos_encoder_forward(rel_pos)
                + self.rel_pos_encoder_backward(rel_pos.transpose(-2, -1))
            ).permute(0, 3, 1, 2)
        else:
            rel_pos_bias = None
        # x: (b, n_head, l_q, d_k), attn: (b, n_head, l_q, l_k)
        x = attention(query, key, value, self.attn_dropout, rel_pos_bias)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.resid_dropout(self.proj(x))


# Different Attention Blocks, All Based on MultiHeadAttention
class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float,
        droppath: float,
        rel_pos_bias: bool = False,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # The larger the dataset, the better rel_pos_bias works
        # probably due to the overfitting of rel_pos_bias
        self.attn = MultiHeadAttention(
            dim, n_head, dropout, q_learnable=False, rel_pos_bias=rel_pos_bias
        )
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()

    def forward(self, x: Tensor, rel_pos: Optional[Tensor]) -> Tensor:
        x_ = self.norm(x)
        x_ = self.attn(x_, x_, x_, rel_pos)
        return self.drop_path(x_) + x


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_head: int, dropout: float, droppath: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, n_head, dropout, q_learnable=True)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()

    def forward(self, x: Tensor, learnt_q: Tensor) -> Tensor:
        x_ = self.norm(x)
        x_ = self.attn(learnt_q, x_, x_)
        # In multi_stage' attention, no residual connection is used because of the change in output shape
        return self.drop_path(x_)


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: Optional[int] = None,
        act_layer: str = "relu",
        drop: float = 0.0,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        elif act_layer.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class GCNMlp(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: Optional[int] = None,
        act_layer: str = "relu",
        drop: float = 0.0,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.gcn = nn.Linear(in_features, hidden_features)
        if act_layer.lower() == "relu":
            self.act = nn.ReLU()
        elif act_layer.lower() == "leaky_relu":
            self.act = nn.LeakyReLU()
        elif act_layer.lower() == "gelu":
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        x1 = self.fc1(x)
        gcn_x1, gcn_x2 = self.gcn(x).chunk(2, dim=-1)
        x = x1 + torch.cat([adj @ gcn_x1, adj.mT @ gcn_x2], dim=-1)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float,
        act_layer: str,
        dropout: float,
        droppath: float,
        gcn: bool = False,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        if gcn:
            self.feed_forward = GCNMlp(
                dim, mlp_ratio, act_layer=act_layer, drop=dropout
            )
        else:
            self.feed_forward = Mlp(dim, mlp_ratio, act_layer=act_layer, drop=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        x_ = self.norm(x)
        x_ = self.feed_forward(x_, adj)
        return self.drop_path(x_) + x


class EncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        mlp_ratio: float,
        act_layer: str,
        dropout: float,
        droppath: float,
    ):
        super().__init__()
        self.self_attn = SelfAttentionBlock(
            dim, n_head, dropout, droppath, rel_pos_bias=True
        )
        self.feed_forward = FeedForwardBlock(
            dim, mlp_ratio, act_layer, dropout, droppath, gcn=True
        )

    def forward(self, x: Tensor, rel_pos: Tensor, adj: Tensor) -> Tensor:
        x = self.self_attn(x, rel_pos)
        x = self.feed_forward(x, adj)
        return x


# Blocks Used in Encoder
class FuseFeatureBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float,
        droppath: float,
        mlp_ratio: float = 4.0,
        act_layer: str = "relu",
    ):
        super().__init__()
        self.norm_kv = nn.LayerNorm(dim)
        self.norm_q = nn.LayerNorm(dim)
        self.fuse_attn = MultiHeadAttention(dim, n_head, dropout, q_learnable=False)
        self.feed_forward = FeedForwardBlock(
            dim, mlp_ratio, act_layer, dropout, droppath
        )

    def forward(self, memory: Tensor, q: Tensor) -> Tensor:
        x_ = self.norm_kv(memory)
        q_ = self.norm_q(q)
        x = self.fuse_attn(q_, x_, x_)
        x = self.feed_forward(x)
        return x


class FuseStageBlock(nn.Module):
    def __init__(
        self,
        depths: List[int],
        dim: int,
        n_head: int,
        mlp_ratio: float,
        act_layer: str,
        dropout: float,
        droppath: float,
        stg_id: int,
        dp_rates: float,
    ):
        super().__init__()
        self.n_self_attn = depths[stg_id] - 1
        self.self_attns = nn.ModuleList()
        self.feed_forwards = nn.ModuleList()
        for i, droppath in enumerate(dp_rates):
            if i == 0:
                self.cross_attn = CrossAttentionBlock(dim, n_head, dropout, droppath)
            else:
                self.self_attns.append(
                    SelfAttentionBlock(dim, n_head, dropout, droppath)
                )
            self.feed_forwards.append(
                FeedForwardBlock(dim, mlp_ratio, act_layer, dropout, droppath)
            )

    def forward(self, kv: Tensor, q: Tensor) -> Tensor:
        x = self.cross_attn(kv, q)
        x = self.feed_forwards[0](x)
        for i in range(self.n_self_attn):
            x = self.self_attns[i](x)
            x = self.feed_forwards[i + 1](x)
        return x


# Main class
class Encoder(nn.Module):
    def __init__(
        self,
        depths: List[int] = [6, 1, 1, 1],
        dim: int = 192,
        n_head: int = 6,
        mlp_ratio: float = 4.0,
        act_layer: str = "relu",
        dropout: float = 0.1,
        droppath: float = 0.0,
    ):
        super().__init__()
        self.num_stage = len(depths)
        self.num_layers = sum(depths)
        self.norm = nn.LayerNorm(dim)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, droppath, self.num_layers)]

        # 1st stage: Encoder
        self.layers = nn.ModuleList()
        for i in range(depths[0]):
            droppath = dpr[i]
            self.layers.append(
                EncoderBlock(dim, n_head, mlp_ratio, act_layer, dropout, droppath)
            )

        if self.num_stage > 1:
            # Rest stage: information fusion
            self.fuseUnit = nn.ModuleList()
            self.fuseStages = nn.ModuleList()
            self.fuseStages.append(
                FuseStageBlock(
                    depths,
                    dim,
                    n_head,
                    mlp_ratio,
                    act_layer,
                    dropout,
                    droppath,
                    stg_id=1,
                    dp_rates=dpr[sum(depths[:1]) : sum(depths[:2])],
                )
            )
            for i in range(2, self.num_stage):
                self.fuseUnit.append(
                    FuseFeatureBlock(
                        dim,
                        n_head,
                        dropout,
                        droppath,
                        mlp_ratio,
                        act_layer,
                    )
                )
                self.fuseStages.append(
                    FuseStageBlock(
                        depths,
                        dim,
                        n_head,
                        mlp_ratio,
                        act_layer,
                        dropout,
                        droppath,
                        stg_id=i,
                        dp_rates=dpr[sum(depths[:i]) : sum(depths[: i + 1])],
                    )
                )

            self.learnt_q = nn.ParameterList(
                [
                    nn.Parameter(torch.randn(1, 2 ** (3 - s), dim))
                    for s in range(1, self.num_stage)
                ]
            )

    def forward(self, x: Tensor, rel_pos: Tensor, adj: Tensor) -> Tensor:
        B, _, _ = x.shape

        # 1st stage: Encoder
        for i, layer in enumerate(self.layers):
            x = layer(x, rel_pos, adj)
        x_ = x
        # Rest stage: information fusion
        if self.num_stage > 1:
            memory = x
            q = self.fuseStages[0](
                memory, self.learnt_q[0].repeat(B, 1, 1, 1)
            )  # q(b,4,d)
            for i in range(self.num_stage - 2):
                kv = self.fuseUnit[i](memory, q)
                q = self.fuseStages[i + 1](
                    kv, self.learnt_q[i + 1].repeat(B, 1, 1, 1)
                )  # q(b,2,d), q(b,1,d)
            x_ = q
        output = self.norm(x_)
        return output


class RegHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        avg_tokens: bool = False,
        out_channels: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.avg_tokens = avg_tokens
        self.layer = nn.Linear(in_channels, out_channels)

    def forward(self, x: Tensor) -> Tensor:  # x(b/n_gpu, l, d)
        if self.avg_tokens:
            x_ = x.mean(dim=1)
        else:
            x_ = x[:, 0, :]  # (b, d)

        res = self.layer(x_)
        return res + 0.5


def tokenizer(ops, adj: Tensor, dim_x=48, dim_p=48, embed_type="nape"):
    adj = torch.tensor(adj)

    # encode operation
    fn = Embedder(dim_x, embed_type=embed_type)
    code_ops_list = [fn(torch.Tensor([30]))]
    code_ops_list += [fn(torch.Tensor([op])) for op in ops]
    code_ops = torch.stack(code_ops_list, dim=0)  # (len, dim_x)

    # encode self position
    code_pos_list = [fn(torch.Tensor([30]))]
    code_pos_list += [fn(torch.Tensor([i])) for i in range(len(ops))]
    code_pos = torch.stack(code_pos_list, dim=0)  # (len, dim_p)
    code = torch.cat([code_ops, code_pos], dim=-1)

    depth = torch.Tensor([len(ops)])
    depth_fn = Embedder(dim_x + dim_p, embed_type=embed_type)
    code_depth = depth_fn(depth).reshape(1, -1)

    shortest_path, path = algos.floyd_warshall(adj.numpy())
    shortest_path = torch.from_numpy(shortest_path).long()
    shortest_path = torch.clamp(shortest_path, min=0, max=8)

    rel_pos = torch.full((len(ops) + 2, len(ops) + 2), fill_value=9).int()
    rel_pos[1:-1, 1:-1] = shortest_path

    return code, rel_pos, code_depth


class NeuralFormer(nn.Module):
    def __init__(
        self,
        depths: List[int] = [6, 1, 1, 1],
        dim: int = 192,
        n_head: int = 6,
        mlp_ratio: float = 4.0,
        act_layer: str = "relu",
        dropout: float = 0.1,
        droppath: float = 0.0,
        avg_tokens: bool = False,
        use_extra_token: bool = True,
        dataset: str = "nasbench",
    ):
        super().__init__()
        self.dim = dim
        self.use_extra_token = use_extra_token

        self.transformer = Encoder(
            depths=depths,
            dim=dim,
            n_head=n_head,
            mlp_ratio=mlp_ratio,
            act_layer=act_layer,
            dropout=dropout,
            droppath=droppath,
        )
        self.head = RegHead(dim, avg_tokens, 1, dropout)
        if use_extra_token:
            self.dep_map = nn.Linear(dim, dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.constant_(m.weight, 0)
            # nn.init.trunc_normal_(m.weight, std=0.02)

    @torch.jit.ignore()
    def no_weight_decay(self):
        no_decay = {}
        # no_decay = {
        #     f"transformer.learnt_q.{i}" for i in range(len(self.config.depths) - 1)
        # }
        return no_decay

    def forward(self, sample, static_feats) -> Tensor:
        # Original Encodings
        seqcode = sample["code"]
        depth = sample["code_depth"]
        rel_pos = sample["code_rel_pos"]
        adj = sample["code_adj"]
        # Depth token
        if self.use_extra_token:
            code_depth = F.relu(self.dep_map(depth))
            seqcode = torch.cat([seqcode, code_depth], dim=1)

        aev = self.transformer(seqcode, rel_pos, adj.to(torch.float))
        # multi_stage:aev(b, 1, d)
        predict = self.head(aev)
        return predict
