import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
from torch import Tensor

# import algos
from neuralformer.data_process.position_encoding import Embedder


class SquareReLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x * F.relu(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        n_head: int,
        dropout: float = 0.0,
        rel_pos_bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.head_size = dim // n_head  # default: 32

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        self.rel_pos_bias = rel_pos_bias
        # if rel_pos_bias:
        #     self.rel_pos_forward = nn.Embedding(10, self.n_head, padding_idx=9)
        #     self.rel_pos_backward = nn.Embedding(10, self.n_head, padding_idx=9)

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        B, L, C = x.shape

        query, key, value = self.qkv(x).chunk(3, -1)
        query = query.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        attn = torch.matmul(query, key.mT) / math.sqrt(self.head_size)
        if self.rel_pos_bias:
            adj = adj.masked_fill(torch.logical_and(adj > 1, adj < 9), 0)
            adj = adj.masked_fill(adj != 0, 1)
            adj = adj.float()
            # pe = torch.stack([adj], dim=1).repeat(1, self.n_head // 1, 1, 1)
            # pe = torch.stack([adj.mT], dim=1).repeat(1, self.n_head // 1, 1, 1)
            # pe = torch.stack([adj, adj.mT], dim=1).repeat(1, self.n_head // 2, 1, 1)
            # pe = torch.stack([adj, adj.mT, adj @ adj, adj.mT @ adj.mT], dim=1)
            pe = torch.stack([adj, adj.mT, adj.mT @ adj, adj @ adj.mT], dim=1)
            pe = pe + torch.eye(L, dtype=adj.dtype, device=adj.device)
            pe = pe.int()

            # pe = (
            #     self.rel_pos_forward(rel_pos) + self.rel_pos_backward(rel_pos.mT)
            # ).permute(0, 3, 1, 2)
            # attn = attn * (1 + pe)
            attn = attn.masked_fill(pe == 0, -torch.inf)
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)  # (b, n_head, l_q, l_k)
        x = torch.matmul(attn, value)

        x = x.transpose(1, 2).contiguous().view(B, L, self.dim)
        return self.resid_dropout(self.proj(x))

    def extra_repr(self) -> str:
        return f"n_head={self.n_head}"


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
        self.attn = MultiHeadAttention(dim, n_head, dropout, rel_pos_bias=rel_pos_bias)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()

    def forward(self, x: Tensor, rel_pos: Optional[Tensor] = None) -> Tensor:
        x_ = self.norm(x)
        x_ = self.attn(x_, rel_pos)
        return self.drop_path(x_) + x


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
        elif act_layer.lower() == "square_relu":
            self.act = SquareReLU()
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
        elif act_layer.lower() == "square_relu":
            self.act = SquareReLU()
        else:
            raise ValueError(f"Unsupported activation: {act_layer}")
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        out = self.fc1(x)
        gcn_x1, gcn_x2 = self.gcn(x).chunk(2, dim=-1)
        out = out + torch.cat([adj @ gcn_x1, adj.mT @ gcn_x2], dim=-1)
        out = self.act(out)
        out = self.drop1(out)
        out = self.fc2(out)
        out = self.drop2(out)
        return out


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
            self.mlp = GCNMlp(dim, mlp_ratio, act_layer=act_layer, drop=dropout)
        else:
            self.mlp = Mlp(dim, mlp_ratio, act_layer=act_layer, drop=dropout)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()

    def forward(self, x: Tensor, adj: Optional[Tensor] = None) -> Tensor:
        x_ = self.norm(x)
        x_ = self.mlp(x_, adj)
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
            dim, n_head, dropout, droppath, rel_pos_bias=False
        )
        self.feed_forward = FeedForwardBlock(
            dim, mlp_ratio, act_layer, dropout, droppath, gcn=True
        )

    def forward(self, x: Tensor, rel_pos: Tensor, adj: Tensor) -> Tensor:
        x = self.self_attn(x, rel_pos)
        x = self.feed_forward(x, adj)
        return x


# Main class
class Encoder(nn.Module):
    def __init__(
        self,
        depths: List[int] = [12],
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

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, droppath, self.num_layers)]

        # Encoder stage
        self.layers = nn.ModuleList()
        for i in range(depths[0]):
            droppath = dpr[i]
            self.layers.append(
                EncoderBlock(dim, n_head, mlp_ratio, act_layer, dropout, droppath)
            )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, rel_pos: Tensor, adj: Tensor) -> Tensor:
        # 1st stage: Encoder
        for i, layer in enumerate(self.layers):
            x = layer(x, rel_pos, adj)
        output = self.norm(x)
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
        return res


def tokenizer(
    ops: List[int], adj, depth: int, dim_x: int = 192, embed_type: str = "nape"
):
    adj = torch.tensor(adj)

    if embed_type == "onehot_op":
        code_ops = F.one_hot(torch.tensor(ops), num_classes=dim_x)
        code_depth = F.one_hot(torch.tensor([depth]), num_classes=dim_x)
        return (code_ops.to(torch.int8), adj.to(torch.int8), code_depth.to(torch.int8))
    elif embed_type == "onehot_oppos":
        code_ops = F.one_hot(torch.tensor(ops), num_classes=dim_x // 2)
        code_pos = F.one_hot(torch.arange(len(ops)), num_classes=dim_x // 2)
        code_ops = torch.cat([code_ops, code_pos], dim=-1)
        # # Another implementation
        # code_ops = F.one_hot(torch.tensor(ops), num_classes=dim_x)
        # code_ops[:, dim_x // 2:] = F.one_hot(torch.arrange(len(ops)), num_classes=dim_x // 2)
        code_depth = F.one_hot(torch.tensor([depth]), num_classes=dim_x)
        return (code_ops.to(torch.int8), adj.to(torch.int8), code_depth.to(torch.int8))
    elif embed_type == "onehot_oplaplacian":
        code_ops = F.one_hot(torch.tensor(ops), num_classes=dim_x)
        code_ops[:, dim_x // 2 : dim_x // 2 + adj.shape[-1]] = adj.sum(-1) - adj
        code_depth = F.one_hot(torch.tensor([depth]), num_classes=dim_x)
        return (code_ops.to(torch.int8), adj.to(torch.int8), code_depth.to(torch.int8))
    else:
        # encode operation
        fn = Embedder(dim_x // 2, embed_type=embed_type)
        code_ops_list = [fn(torch.Tensor([30]))]
        code_ops_list += [fn(torch.Tensor([op])) for op in ops]
        code_ops = torch.stack(code_ops_list, dim=0)  # (len, dim_x)

        depth = torch.Tensor([depth])
        code_depth = fn(depth).reshape(1, -1)

        # shortest_path, path = algos.floyd_warshall(adj.numpy())
        # shortest_path = torch.from_numpy(shortest_path).long()
        # shortest_path = torch.clamp(shortest_path, min=0, max=8)

        rel_pos = torch.full((len(ops) + 2, len(ops) + 2), fill_value=9).int()
        rel_pos[1:-1, 1:-1] = adj
        # rel_pos[0, 0] = 0
        # rel_pos[-1, -1] = 0
        return code_ops, rel_pos, code_depth


class NeuralFormer(nn.Module):
    def __init__(
        self,
        depths: List[int] = [12],
        in_chans: int = 32,
        dim: int = 192,
        n_head: int = 4,
        mlp_ratio: float = 4.0,
        act_layer: str = "relu",
        dropout: float = 0.1,
        droppath: float = 0.0,
        avg_tokens: bool = False,
        class_token: bool = True,
        depth_embed: bool = True,
        dataset: str = "nasbench",
    ):
        super().__init__()

        self.op_embed = nn.Linear(in_chans, dim, False)
        self.depth_embed = nn.Linear(in_chans, dim, False) if depth_embed else None
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim)) if class_token else None

        self.norm = nn.LayerNorm(dim)
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

        self.init_weights()

    def init_weights(self):
        self.apply(self._init_weights)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

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
        return no_decay

    def forward(self, sample, static_feats) -> Tensor:
        # Original Encodings
        seqcode = sample["code"]
        depth = sample["code_depth"]
        rel_pos = sample["code_rel_pos"]
        adj = sample["code_adj"]

        seqcode = self.op_embed(seqcode)
        if self.cls_token is not None:
            seqcode = torch.cat(
                [self.cls_token.expand(seqcode.shape[0], -1, -1), seqcode], dim=1
            )
            new_adj = torch.zeros(
                adj.shape[0], adj.shape[1] + 1, adj.shape[2] + 1, device=adj.device
            )
            new_adj[:, 1:, 1:] = adj
            adj = new_adj
        if self.depth_embed is not None:
            seqcode = torch.cat([seqcode, self.depth_embed(depth)], dim=1)
            new_adj = torch.zeros(
                adj.shape[0], adj.shape[1] + 1, adj.shape[2] + 1, device=adj.device
            )
            new_adj[:, :-1, :-1] = adj
            adj = new_adj
        seqcode = self.norm(seqcode)

        aev = self.transformer(seqcode, rel_pos, adj.to(torch.float))
        # multi_stage:aev(b, 1, d)
        predict = self.head(aev) + 0.5
        return predict
