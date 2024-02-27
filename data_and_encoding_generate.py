import os
import torch
import random
import argparse
import json
import tqdm
from neuralformer.models.encoders import tokenizer


def argLoader():
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
        "--multires_x", type=int, default=96, help="dim of operation encoding"
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        default="trans",
        help="Type of position embedding: nape|nerf|trans",
    )
    parser.add_argument("--split_type", type=str, default="GATES", help="GATES|TNASP")
    args = parser.parse_args()
    return args


def get_nasbench101_item(archs, i: int):
    index = str(i)
    ops = archs[index]["module_operations"]
    adj = archs[index]["module_adjacency"]
    depth = len(ops)
    code, rel_pos, code_depth = tokenizer(
        ops,
        adj,
        depth,
        args.multires_x,
        args.embed_type,
    )
    return {
        "index": i,
        "adj": adj,
        "ops": ops,
        "validation_accuracy": archs[index]["validation_accuracy"],
        "test_accuracy": archs[index]["test_accuracy"],
        "code": code,
        "code_rel_pos": rel_pos,
        "code_depth": code_depth,
    }


def get_nasbench201_item(archs, i: int):
    index = str(i)
    ops = archs[index]["module_operations"]
    adj = archs[index]["module_adjacency"]
    depth = len([op for op in ops if op != 5])  # `op == 5` indicates `none`
    code, rel_pos, code_depth = tokenizer(
        ops,
        adj,
        depth,
        args.multires_x,
        args.embed_type,
    )
    return {
        "index": i,
        "adj": adj,
        "ops": ops,
        "test_accuracy": archs[index]["test_accuracy"],
        "test_accuracy_avg": archs[index]["test_accuracy_avg"],
        "valid_accuracy": archs[index]["validation_accuracy"],
        "valid_accuracy_avg": archs[index]["validation_accuracy_avg"],
        "code": code,
        "code_rel_pos": rel_pos,
        "code_depth": code_depth,
    }


if __name__ == "__main__":
    args = argLoader()

    random.seed(args.seed)
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.load_all:
        with open(args.data_path) as f:
            archs = json.load(f)

        id_list = list(range(0, len(archs)))
        # random.shuffle(id_list)

        data = {}
        for i in tqdm.trange(len(archs)):
            if args.dataset == "nasbench101":
                data[id_list[i]] = get_nasbench101_item(archs, i)
            elif args.dataset == "nasbench201":
                data[id_list[i]] = get_nasbench201_item(archs, i)
        torch.save(data, os.path.join(save_dir, f"all_{args.dataset}.pt"))

    if not args.load_all:
        if args.dataset == "nasbench101":
            torch.set_num_threads(1)
            train_data = {}
            test_data = {}
            val_data = {}
            with open(args.data_path) as f:
                archs = json.load(f)
            print(len(archs))
            id_list = list(range(0, len(archs)))
            # Split dataset following TNASP
            if args.split_type == "TNASP":
                random.shuffle(id_list)
                train_list = id_list
                l1 = int(len(archs) * args.n_percent)
                lv = int(len(archs) * (args.n_percent + 0.0005))
                l2 = int(len(archs) * (args.n_percent + 0.0005))  # val 0.05%

            # Split dataset following GATES
            if args.split_type == "GATES":
                train_list = id_list[: int(len(archs) * 0.9)]
                random.shuffle(train_list)
                l1 = int(len(archs) * 0.9 * args.n_percent)
                lv = int(len(archs) * (0.9 * args.n_percent + 0.0005))
                l2 = int(len(archs) * 0.9)

            for i in train_list[:l1]:
                idx = len(train_data)
                train_data[idx] = get_nasbench101_item(archs, i)
            torch.save(train_data, os.path.join(save_dir, "train.pt"))

            # for i in id_list[l1:l2]:
            for i in train_list[l1:lv]:
                idx = len(val_data)
                val_data[idx] = get_nasbench101_item(archs, i)
            torch.save(val_data, os.path.join(save_dir, "val.pt"))

            for i in id_list[l2:]:
                idx = len(test_data)
                test_data[idx] = get_nasbench101_item(archs, i)
            torch.save(test_data, os.path.join(save_dir, "test.pt"))

        elif args.dataset == "nasbench201":
            with open(args.data_path) as f:
                archs = json.load(f)
            print(len(archs))
            id_list = list(range(0, len(archs)))
            # Split dataset following TNASP
            if args.split_type == "TNASP":
                random.shuffle(id_list)
                train_list = id_list
                l1 = int(len(archs) * args.n_percent)
                lv = int(len(archs) * args.n_percent) + 200
                l2 = int(len(archs) * args.n_percent) + 200

            # Split dataset following GATES
            if args.split_type == "GATES":
                train_list = id_list[: int(len(archs) * 0.5)]
                random.shuffle(train_list)
                l1 = int(len(archs) * 0.5 * args.n_percent)
                # lv = int(len(archs)*(0.9*args.n_percent + 0.0005))
                l2 = int(len(archs) * 0.5)

            train_data, test_data = {}, {}
            for i in train_list[:l1]:
                idx = len(train_data)
                train_data[idx] = get_nasbench201_item(archs, i)
            torch.save(train_data, os.path.join(save_dir, "train.pt"))

            if args.split_type == "TNASP":
                val_data = {}
                for i in train_list[l1:lv]:
                    print(i)
                    idx = len(val_data)
                    val_data[idx] = get_nasbench201_item(archs, i)
                torch.save(val_data, os.path.join(save_dir, "val.pt"))

            for i in id_list[l2:]:
                print(i)
                idx = len(test_data)
                test_data[idx] = get_nasbench201_item(archs, i)
            torch.save(test_data, os.path.join(save_dir, "test.pt"))
