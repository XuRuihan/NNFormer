import argparse
import os
import random

import torch
from torch_geometric.utils import to_dense_adj

from nnformer.models.encoders import tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="nasbench101", help="dataset type"
    )
    parser.add_argument(
        "--data_path", type=str, default="nasbench101.json", help="path of json file"
    )
    parser.add_argument(
        "--save_dir", type=str, default=".", help="path of generated pt files"
    )
    parser.add_argument(
        "--n_percent", type=float, default=0.01, help="train proportion"
    )
    parser.add_argument(
        "--load_all", type=bool, default=True, help="load total dataset"
    )
    parser.add_argument(
        "--enc_dim", type=int, default=32, help="dim of operation encoding"
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        default="onehot_op",
        help="Type of position embedding: onehot_op|onehot_oppos|nape|nerf|trans",
    )
    parser.add_argument("--split_type", type=str, default="GATES", help="GATES|TNASP")
    args = parser.parse_args()
    return args


def pyg2dict(pyg_data, index, enc_dim, embed_type):
    adj = to_dense_adj(pyg_data.edge_index).squeeze(0).tolist()
    ops = pyg_data.x.nonzero(as_tuple=False)[:, 1].tolist()

    depth = len(ops)
    code, rel_pos, code_depth = tokenizer(ops, adj, depth, enc_dim, embed_type)

    dict_data = {
        "index": index,
        "adj": adj,
        "ops": ops,
        "validation_accuracy": pyg_data.y.item(),
        "code": code,
        "code_rel_pos": rel_pos,
        "code_depth": code_depth,
    }
    return dict_data


def main():
    args = parse_args()

    random.seed(args.seed)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = torch.load(args.data_path)
    data = [pyg2dict(d, i, args.enc_dim, args.embed_type) for i, d in enumerate(data)]
    # num_train = int(0.5 * len(data))
    # train_data, test_data = data[:num_train], data[num_train:]
    # val_data, test_data = test_data[:40], test_data[40:]
    # train_data = train_data[: int(num_train * args.n_percent)]

    if args.load_all:
        torch.save(data, os.path.join(save_dir, f"gates_{args.dataset}.pt"))


if __name__ == "__main__":
    main()
