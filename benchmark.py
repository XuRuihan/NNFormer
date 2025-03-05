# import os

import torch.utils.benchmark as benchmark

# from config import parse_args
# from nnformer.data_process import init_dataloader
from nnformer.models.encoders import NNFormer

# from nnformer.utils import setup_logger
import torch

# args = parse_args()
# logger = setup_logger("output/benchmark.log")

net = NNFormer()

# train_loader, val_loader = init_dataloader(args, logger)

# x = train_loader.dataset[0][0]


seqcode = torch.rand(1, 7, 32)
depth = torch.rand(1, 1, 32)
rel_pos = torch.rand(1, 8, 8)
adj = torch.rand(1, 8, 8)

sample = dict()
sample["code"] = seqcode
sample["code_depth"] = depth
sample["code_rel_pos"] = rel_pos
sample["code_adj"] = adj


t0 = benchmark.Timer(
    stmt="net(x, None)",
    setup="from __main__ import net",
    globals={"x": sample},
)

print(t0.timeit(100))
