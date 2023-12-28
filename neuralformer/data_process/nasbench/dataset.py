import torch
from torch.utils.data import Dataset

import random
import time


class NasbenchDataset(Dataset):
    def __init__(
        self,
        logger,
        dataset,
        part,
        data_path,
        percent=0,
        lbd_consistency=0,
    ):
        self.part = part
        self.dataset = dataset
        self.consistency = lbd_consistency
        self.data_path = data_path
        self.percent = percent

        t0 = time.time()
        logger.info(f"Building dataset {self.part} from .pth file")
        self.data = self._load()
        t = time.time() - t0
        if self.consistency > 0:
            total_samples = sum(len(data) for data in self.data)
            logger.info(
                f"Finish Loading dataset {self.part}; Number of sample groups: {len(self.data)}; "
                f"Total samples: {total_samples}; Time: {t:.2f}s"
            )
        else:
            logger.info(
                f"Finish Loading dataset {self.part}; Number of samples: {len(self.data)}; Time: {t:.2f}s"
            )

    def _load(self):
        data_file = self.data_path
        datas = [torch.load(data_file)]

        loaded_data = []
        data_num = (
            int(self.percent) if self.percent > 1 else int(len(datas[0]) * self.percent)
        )
        if self.part == "train":
            keys = list(range(data_num))
        elif self.part == "val":
            keys = list(range(data_num, data_num + 1024))
        elif self.part == "test":
            keys = list(range(len(datas[0])))  # test all
            # keys = list(range(data_num + 1024, data_num+2048)) # test 1024
            # keys = list(range(data_num + 1024, len(datas[0]))) # test rest

        for key in keys:
            # use consistency loss
            if self.part == "train" and self.consistency > 0:
                loaded_data.append([datas[0][key], datas[0][key]])

            # no consistency loss, val and test
            elif (
                (self.part == "train" and self.consistency == 0.0)
                or self.part == "val"
                or self.part == "test"
            ):
                loaded_data.append(datas[0][key])

        return loaded_data

    def __getitem__(self, index):
        if (self.part == "train" and self.consistency > 0):
            # data_0, data_1 = random.sample(self.data[index], 2)
            data_0 = self.data[index][0]
            data_1 = random.sample(self.data[index][1:], 1)[0]
            if self.dataset == "nasbench101":
                data_0, data_1 = self.preprocess_101(data_0), self.preprocess_101(
                    data_1
                )
            elif self.dataset == "nasbench201":
                data_0, data_1 = self.preprocess_201(data_0), self.preprocess_201(
                    data_1
                )
            return data_0, data_1
        else:
            if self.dataset == "nasbench101":
                return self.preprocess_101(self.data[index])
            elif self.dataset == "nasbench201":
                return self.preprocess_201(self.data[index])

    def preprocess_101(self, data):
        ops = torch.IntTensor(data["ops"])
        code = torch.Tensor(data["code"])
        code_rel_pos = torch.IntTensor(data["code_rel_pos"])
        code_depth = torch.Tensor(data["code_depth"])
        val_acc_avg = torch.Tensor([data["validation_accuracy"]])
        test_acc_avg = torch.Tensor([data["test_accuracy"]])
        return {
            "ops": ops,
            "code": code,
            "code_rel_pos": code_rel_pos,
            "code_adj": (code_rel_pos == 1).float(),
            "code_depth": code_depth,
            "val_acc_avg": val_acc_avg,
            "test_acc_avg": test_acc_avg,
        }

    def preprocess_201(self, data):
        ops = torch.IntTensor(data["ops"])
        code = torch.Tensor(data["code"])
        code_rel_pos = torch.IntTensor(data["code_rel_pos"])
        code_depth = torch.Tensor(data["code_depth"])
        val_acc_avg = torch.Tensor([data["valid_accuracy_avg"]]) * 0.01
        test_acc_avg = torch.Tensor([data["test_accuracy_avg"]]) * 0.01
        return {
            "ops": ops,
            "code": code,
            "code_rel_pos": code_rel_pos,
            "code_adj": (code_rel_pos == 1).float(),
            "code_depth": code_depth,
            "val_acc_avg": val_acc_avg,
            "test_acc_avg": test_acc_avg,
        }

    def __len__(self):
        return len(self.data)
