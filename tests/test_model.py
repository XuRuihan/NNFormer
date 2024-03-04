import unittest
import torch
import os
import sys

sys.path.append(os.getcwd())
from neuralformer.models.encoders.neuralformer import NeuralFormer, tokenizer
import neuralformer


class TestModel(unittest.TestCase):
    @torch.inference_mode()
    def test_neuralformer(self):
        model = NeuralFormer()
        model.eval()

        ops = [0, 1, 2, 3, 4]
        adj = [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
        depth = 5
        code, code_rel_pos, code_depth = tokenizer(ops, adj, depth, dim_x=32)
        sample = dict()
        sample["code"] = code.unsqueeze(0)
        sample["code_depth"] = code_depth.unsqueeze(0)
        sample["code_rel_pos"] = code_rel_pos.unsqueeze(0)
        sample["code_adj"] = (code_rel_pos == 1).float()
        y_1 = model(sample, None)
        print(y_1)

        ops = [0, 2, 1, 3, 4]
        adj = [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
        ]
        code, code_rel_pos, code_depth = tokenizer(ops, adj, depth, dim_x=96)
        sample = dict()
        sample["code"] = code.unsqueeze(0)
        sample["code_depth"] = code_depth.unsqueeze(0)
        sample["code_rel_pos"] = code_rel_pos.unsqueeze(0)
        sample["code_adj"] = (code_rel_pos == 1).float()
        y_2 = model(sample, None)
        print(y_2)


if __name__ == "__main__":
    unittest.main()
