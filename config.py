import argparse


def argLoader():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument(
        "--dataset",
        type=str,
        default="nasbench101",
        help="nasbench101||nasbench201||nnlqp",
    )

    # DataLoader
    parser.add_argument(
        "--percent",
        type=float,
        default=4236,
        help="trainings samples, percent or numbers",
    )
    parser.add_argument("--data_path", type=str, default="data/nasbench101/")
    # NNLQP
    parser.add_argument("--override_data", action="store_true")
    parser.add_argument("--test_model_type", type=str, default="resnet18")
    parser.add_argument("--finetuning", type=bool, default=False)

    # Overall
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--drop_path_rate", type=float, default=0.0)

    # Network Settings
    # Encoding
    parser.add_argument(
        "--embed_type",
        type=str,
        default="nape",
        help="Type of position embedding: nape|nerf|trans",
    )
    parser.add_argument("--depths", nargs="+", type=int, default=[6, 1, 1, 1])
    parser.add_argument(
        "--act_function",
        type=str,
        default="relu",
        help="activation function used in transformer",
    )
    parser.add_argument(
        "--use_extra_token",
        action="store_true",
        help="Whether use a extra token to predict",
    )
    parser.add_argument(
        "--multires_x",
        type=int,
        default=96,
        help="Operations encoding dim = 2*multires_x",
    )
    parser.add_argument("--graph_d_model", type=int, default=192)
    parser.add_argument("--graph_n_head", type=int, default=6)
    parser.add_argument("--graph_d_ff", type=int, default=768)

    # Head
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument(
        "--avg_tokens",
        type=bool,
        default=False,
        help="Whether average the tokens of embedding to predict",
    )

    # Device
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n_workers", type=int, default=8)

    # # Optimizer
    # parser.add_argument("--lr", type=float, default=1e-3)
    # parser.add_argument("--weight_decay", type=float, default=0.01)
    # parser.add_argument("--adam_epsilon", type=float, default=1e-8)

    # Optimizer parameters
    group = parser.add_argument_group("Optimizer parameters")
    group.add_argument(
        "--opt",
        default="adamw",
        type=str,
        metavar="OPTIMIZER",
        help='Optimizer (default: "adamw")',
    )
    group.add_argument(
        "--opt-eps",
        default=None,
        type=float,
        metavar="EPSILON",
        help="Optimizer Epsilon (default: None, use opt default)",
    )
    group.add_argument(
        "--opt-betas",
        default=None,
        type=float,
        nargs="+",
        metavar="BETA",
        help="Optimizer Betas (default: None, use opt default)",
    )
    group.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        metavar="M",
        help="Optimizer momentum (default: 0.9)",
    )
    group.add_argument(
        "--weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
    )

    # Learning rate schedule parameters
    group = parser.add_argument_group("Learning rate schedule parameters")
    group.add_argument(
        "--sched",
        default="cosine",
        type=str,
        metavar="SCHEDULER",
        help='LR scheduler (default: "cosine")',
    )
    group.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="learning rate (default: 0.05)",
    )
    group.add_argument(
        "--lr_cycle_mul",
        type=float,
        default=1.0,
        metavar="MULT",
        help="learning rate cycle len multiplier (default: 1.0)",
    )
    group.add_argument(
        "--min_ratio",
        type=float,
        default=1e-1,
        help="lower lr bound for cyclic schedulers that hit 0 (1e-1)",
    )
    group.add_argument(
        "--decay_rate",
        "--dr",
        type=float,
        default=0.1,
        metavar="RATE",
        help="LR decay rate (default: 0.1)",
    )
    group.add_argument(
        "--warmup-lr",
        type=float,
        default=1e-6,
        metavar="LR",
        help="warmup learning rate (default: 1e-6)",
    )
    group.add_argument(
        "--warmup-epochs",
        type=int,
        default=5,
        metavar="N",
        help="epochs to warmup LR, if scheduler supports",
    )
    group.add_argument(
        "--lr-cycle-limit",
        type=int,
        default=1,
        metavar="N",
        help="learning rate cycle limit, cycles enabled if > 1",
    )

    group.add_argument(
        "--epochs",
        type=int,
        default=4000,
        metavar="N",
        help="number of epochs to train (default: 4000)",
    )

    # Training Parameters
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_path", type=str, default="model/")
    parser.add_argument("--save_epoch_freq", type=int, default=1000)
    parser.add_argument("--pretrained_path", type=str, default=None)  # test

    # EMA
    parser.add_argument("--model_ema", action="store_true")
    parser.add_argument("--model_ema_decay", type=float, default=0.99)
    parser.add_argument("--model_ema_force_cpu", type=bool, default=True, help="")
    parser.add_argument(
        "--model_ema_eval",
        type=bool,
        default=True,
        help="Using ema to eval during training.",
    )

    # Loss Parameters
    parser.add_argument(
        "--lambda_mse",
        type=float,
        default=1.0,
        help='weight of mse loss (default: "1.0")',
    )
    parser.add_argument(
        "--lambda_diff",
        type=float,
        default=0.2,
        help='weight of diff loss (default: "0.2")',
    )
    parser.add_argument(
        "--lambda_consistency", type=float, default=0, help="weight of consistency loss"
    )

    # Evalutation
    parser.add_argument(
        "--test_freq", type=int, default=1, help='Test frequency (default: "1")'
    )

    args = parser.parse_args()

    return args
