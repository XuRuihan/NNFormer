import torch
from timm.utils import ModelEma
from timm.optim import create_optimizer_v2, optimizer_kwargs

from neuralformer.optim.scheduler import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup
)
from neuralformer.models.encoders import NeuralFormer
from neuralformer.models.losses import NARLoss
from .utils import model_info
from parallel import DataParallelModel, DataParallelCriterion


def init_layers(args, logger):
    # Model
    net = NeuralFormer(
        depths=args.depths,
        dim=args.graph_d_model,
        n_head=args.graph_n_head,
        mlp_ratio=args.graph_d_ff // args.graph_d_model,
        act_layer=args.act_function,
        dropout=args.dropout,
        droppath=args.drop_path_rate,
        avg_tokens=args.avg_tokens,
        use_extra_token=args.use_extra_token,
        dataset=args.dataset
    )

    if not args.do_train:
        return net

    loss = NARLoss()
    model_info(net, logger)

    if torch.cuda.is_available():
        net = net.cuda(args.device)
        loss = loss.cuda(args.device)
        if args.parallel:
            net = DataParallelModel(net)
            loss = DataParallelCriterion(loss)

    # Model EMA
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEma(
            net, decay=args.model_ema_decay, device=args.device, resume=""
        )
        logger.info(f"Using EMA with decay = {args.model_ema_decay:.8f}")
    else:
        model_ema = None

    return net, model_ema, loss


def init_optim(args, net, nbatches, warm_step=0.1):
    optimizer = create_optimizer_v2(net, **optimizer_kwargs(args))

    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warm_step * nbatches * args.epochs,
        num_training_steps=nbatches * args.epochs,
        num_cycles=1,
        min_ratio=args.min_ratio
    )
    print("warmup steps:", warm_step * nbatches * args.epochs)
    return optimizer, lr_scheduler


def auto_load_model(args, model, model_ema=None, optimizer=None, scheduler=None):
    if args.do_train:
        if args.resume:
            if args.resume.startswith("https"):
                checkpoint = torch.hub.load_state_dict_from_url(
                    args.resume, map_location="cpu", check_hash=True
                )
            else:
                checkpoint = torch.load(args.resume, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            print("Resume checkpoint %s" % args.resume)
            if hasattr(args, "model_ema") and args.model_ema:
                if "model_ema" in checkpoint.keys():
                    model_ema.ema.load_state_dict(checkpoint["state_dict_ema"])
                else:
                    pretrained_dict = {
                        key.replace("module.", ""): value
                        for key, value in checkpoint["state_dict"].items()
                    }
                    model_ema.ema.load_state_dict(pretrained_dict)
            if args.finetuning:
                print("Start fine-tuning from 0-th epoch/iter!")
                return 0
            else:
                if "optimizer" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                    scheduler.load_state_dict(checkpoint["scheduler"])
                print("With optim & ached!")
                if "epoch" in checkpoint.keys():
                    start_id = checkpoint["epoch"]
                elif "iter" in checkpoint.keys():
                    start_id = checkpoint["iter"]
        else:
            start_id = 0
        print("Start training from %d-th epoch/iter!" % (start_id))
        return start_id

    if args.pretrained_path:
        checkpoint = torch.load(
            args.pretrained_path, map_location="cuda:{}".format(args.device)
        )
        if "state_dict_ema" in checkpoint.keys():
            model.load_state_dict(checkpoint["state_dict_ema"])
        else:
            pretrained_dict = {
                key.replace("module.", ""): value
                for key, value in checkpoint["state_dict"].items()
            }
            model.load_state_dict(pretrained_dict)
        print(
            torch.load(
                args.pretrained_path, map_location="cuda:{}".format(args.device)
            )["config"]
        )
