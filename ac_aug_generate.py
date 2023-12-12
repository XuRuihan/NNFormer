import os
import torch
import argparse
import itertools
import numpy as np
from tqdm import trange
from neuralformer.models.encoders import tokenizer


def argLoader():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="nasbench101", help="nasbench101/nasbench301/oo"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../all.pt",
        help="train data needed to be augmented",
    )
    parser.add_argument(
        "--n_percent", type=float, default=1, help="train proportion or numbers"
    )
    parser.add_argument(
        "--multires_x", type=int, default=48, help="dim of operation encoding"
    )
    parser.add_argument(
        "--multires_p", type=int, default=48, help="dim of position encoding"
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        default="nape",
        help="Type of position embedding: nape|nerf|trans",
    )
    args = parser.parse_args()
    return args


def is_toposort(matrix):
    # a toposort is equivalent to the upper triangle adjacency matrix
    for i in range(len(matrix)):
        for j in range(0, i):
            if matrix[i][j] != 0:
                return False
    return True

    # return np.sum(matrix[np.tril_indices(matrix.shape[0], k=-1)]) == 0


def ac_aug_generate(adj, ops):
    num_vertices = len(ops)
    perms = permutations[num_vertices]
    auged_adjs = [adj]
    auged_opss = [ops]
    adj_array = np.array(adj)
    ops_array = np.array(ops)

    for id, perm in enumerate(perms):
        adj_aug = adj_array[perm][:, perm].astype(int).tolist()
        ops_aug = ops_array[perm].astype(int).tolist()
        if is_toposort(adj_aug) and (
            (adj_aug not in auged_adjs) or (ops_aug not in auged_opss)
        ):
            auged_adjs.append(adj_aug)
            auged_opss.append(ops_aug)
    return auged_adjs[1:], auged_opss[1:]


# Pre-calculate permutation sequences
def permutation_sequences():
    permutations = {}
    for num_vertices in range(2, 9):
        temp = list(range(1, num_vertices - 1))
        temp_list = itertools.permutations(temp)

        perms = []
        for id, perm in enumerate(temp_list):
            # skip the identical permutation
            if id == 0:
                continue
            # Keep the first and the last position fixed
            perm = [0] + list(perm) + [num_vertices - 1]
            perms.append(np.array(perm))

        permutations[num_vertices] = perms
    return permutations


if __name__ == "__main__":
    args = argLoader()
    save_dir, file_name = os.path.split(args.data_path)

    dx, dp = args.multires_x, args.multires_p

    train_data = torch.load(args.data_path)

    permutations = permutation_sequences()

    data_num = (
        int(args.n_percent)
        if args.n_percent > 1
        else int(len(train_data) * args.n_percent)
    )
    auged_data = {}
    aug_num = 0
    for key in trange(data_num):
        ops = train_data[key]["ops"]
        adj = train_data[key]["adj"]
        auged_adjs, auged_opss = ac_aug_generate(adj, ops)
        if len(auged_opss) == 0:
            continue
        aug_num += len(auged_opss)
        netcodes = [
            tokenizer(auged_ops, auged_adj, dx, dp, args.embed_type)
            for auged_ops, auged_adj in zip(auged_opss, auged_adjs)
        ]
        auged_data[key] = dict()
        if args.dataset == "nasbench101":
            for i in range(len(auged_opss)):
                auged_data[key][i + 1] = {  # auged data's key : 1 ~ num_auged
                    "index": i,
                    "adj": auged_adjs[i],
                    "ops": auged_opss[i],
                    "validation_accuracy": train_data[key]["validation_accuracy"],
                    "test_accuracy": train_data[key]["test_accuracy"],
                    "code": netcodes[i][0],
                    "code_rel_pos": netcodes[i][1],
                    "code_depth": netcodes[i][2],
                }
        elif args.dataset == "nasbench201":
            for i in range(len(auged_opss)):
                auged_data[key][i + 1] = {
                    "index": i,
                    "adj": auged_adjs[i],
                    "ops": auged_opss[i],
                    "test_accuracy": train_data[key]["test_accuracy"],
                    "test_accuracy_avg": train_data[key]["test_accuracy_avg"],
                    "valid_accuracy": train_data[key]["valid_accuracy"],
                    "valid_accuracy_avg": train_data[key]["valid_accuracy_avg"],
                    "code": netcodes[i][0],
                    "code_rel_pos": netcodes[i][1],
                    "code_depth": netcodes[i][2],
                }
    print(f"Total augment number: {aug_num}")
    torch.save(auged_data, os.path.join(save_dir, f"{data_num}_{args.dataset}_aug.pt"))
