# NeuralFormer: Neural predictor for neural architectures

This is the source code for paper:<br> 
[Predictor-based Neural Architecture Search with Teacher Guidance]()

![NeuralFormer](./assets/neuralformer.png)
Figure 1: **Illustrations of our proposed NeuralFormer.** Our approach uses Neural Architecture Position Encoding (NAPE), Bidirectional Adjacency Aggregation (BAA) in the MLP to enhance local topological features, and Bidirectional Relative Position Embedding (BRPE) in the self-attention layer to introduce the global topological information. NeuralFormer is trained with Teacher Guidance (TG).

## Introduction
Predictor-Based Neural Architecture Search (NAS) leverages predictors to quickly estimate the performance of architectures, thus minimizing the time-consuming training of candidate networks. This approach has gained significant traction and is a prominent branch within NAS. In Predictor-Based NAS algorithms, establishing a robust predictor is the fundamental challenge. Presently, most predictors fall short in robustness and require a substantial number of candidate architecture-performance pairs for effective training. These issues easily lead to undertrained predictors, and contradict the efficiency goal of predictor-based NAS.
We proposed a strong predictor named **NeuralFormer**. By encoding the network topology as features and harnessing the advanced transformer architecture, NeuralFormer achieves promising prediction performance with a relatively small number of training pairs. Additionally, we propose an evolutionary NAS algorithm with teacher guidance, providing more comprehensive knowledge to fully train candidate architectures and boost their performance. Extensive experiments of accuracy prediction and NAS demonstrate that both the proposed NeuralFormer and the teacher-guided evolutionary strategy exhibit impressive performance across various search spaces and vision tasks. 


## Experiments on NAS-Bench-201
We introduce the experimental process using the NAS-Bench-201 dataset as an example. Experiments on NAS-Bench-101 are similar.

### Data preprocessing with our proposed tokenizer
You can generate the preprocessed dataset following the steps below.  This pipeline is similar to [NAR-Former](https://github.com/yuny220/NAR-Former).
1. Download [NAS-Bench-201-v1_0-e61699.pth](https://drive.google.com/file/d/1SKW0Cu0u8-gb18zDpaAGi0f74UdXeGKs/view?pli=1) and put it in `./data/nasbench201/`.

2. Convert the dataset into a JSON file.
   ```
   python preprocessing/gen_json_201.py
   # python preprocessing/gen_json_101.py
   ```
   The generated file `cifar10_valid_converged.json` will be saved in `./data/nasbench201/`.

3. Encode each architecture with our proposed tokenizer.
   ```
   python data_and_encoding_generate.py --dataset nasbench201 --data_path data/nasbench201/cifar10_valid_converged.json --save_dir data/nasbench201/
   # python data_and_encoding_generate.py --dataset nasbench101 --data_path data/nasbench101/nasbench101.json --save_dir data/nasbench101/
   ```
   The generated file `all_nasbench201.pt` will be saved in `./data/nasbench201/`.

### Train NeuralFormer
You can train NeuralFormer following the script below:
```
bash scripts/Accuracy_Predict_NASBench201/train_5%.sh
```
The trained models will be saved in `./output/nasbench-201/neuralformer_5%/`. Training scripts of other settings are shown in the [scripts](./scripts/) directory.

### Evaluate the pretrained model
You can evaluate the trained NeuralFormer following the script below:
```
bash test_5%.sh
```
Evaluating scripts of other models are also shown in the [scripts](./scripts/) directory.

## License
This project is released under the MIT license. Please see the [LICENSE](./LICENSE) file for more information.

<!-- ## Citation
If you find this repository helpful, please consider starring our repo and citing our paper:
```
@article{yi2022nar,
  title={NAR-Former: Neural Architecture Representation Learning towards Holistic Attributes Prediction},
  author={Yi, Yun and Zhang, Haokui and Hu, Wenze and Wang, Nannan and Wang, Xiaoyu},
  journal={arXiv preprint arXiv:2211.08024},
  year={2022}
}
``` -->

## Acknowledgement
This repository is built using the following libraries and repositories.
1. [NAR-Former](https://github.com/yuny220/NAR-Former)
2. [NAS-Bench-101](https://github.com/google-research/nasbench)
3. [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201)
4. [NPENAS](https://github.com/auroua/NPENASv1)
