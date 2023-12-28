import os
import time
import torch
import random
import numpy as np
from timm.utils import get_state_dict

from neuralformer.utils import *
from config import argLoader
from neuralformer.data_process import init_dataloader


def format_second(secs):
    return "{:0>2}:{:0>2}:{:0>2}".format(
        int(secs / 3600), int((secs % 3600) / 60), int(secs % 60)
    )


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(config):
    # Load Dataset
    train_loader, val_loader = init_dataloader(config, logger)
    n_batches = len(train_loader)
    # Init Model
    net, model_ema, criterion = init_layers(config, logger)
    # Optimizer
    optimizer, scheduler = init_optim(config, net, n_batches)
    # Auto Resume
    start_epoch_idx = auto_load_model(config, net, model_ema, optimizer, scheduler)

    # Init Value
    best_tau, best_mape, best_error = -99, 1e5, 0
    if config.model_ema and config.model_ema_eval:
        best_tau_ema, best_mape_ema, best_error_ema = -99, 1e5, 0
    for epoch_idx in range(start_epoch_idx, config.epochs):
        metric = Metric()
        t0 = time.time()

        net.train()
        for batch_idx, batch_data in enumerate(train_loader):
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            if "nasbench" in config.dataset:
                if config.lambda_consistency > 0:
                    data_0, data_1 = batch_data
                    batch_data = {
                        key: torch.cat([data_0[key], data_1[key]], dim=0)
                        for key in data_0.keys()
                    }
                for k, v in batch_data.items():
                    batch_data[k] = v.to(config.device)
                gt = batch_data["val_acc_avg"]
                logits = net(batch_data, None)

            elif config.dataset == "nnlqp":
                codes, gt, sf = (
                    batch_data[0]["netcode"],
                    batch_data[0]["cost"],
                    batch_data[1],
                )
                codes, gt, sf = (
                    codes.to(config.device),
                    gt.to(config.device),
                    sf.to(config.device),
                )
                logits = net(None, None, codes, sf)

            loss_dict = criterion(logits, gt)
            loss = loss_dict["loss"]

            loss.backward()
            optimizer.step()
            scheduler.step()

            if model_ema is not None:
                model_ema.update(net)

            ps = logits.detach().cpu().numpy()[:, 0].tolist()
            gs = gt.detach().cpu().numpy()[:, 0].tolist()
            metric.update(ps, gs)
            acc, err, tau = metric.get()

        t1 = time.time()
        speed = n_batches * args.batch_size / (t1 - t0)
        exp_time = format_second((t1 - t0) * (config.epochs - epoch_idx - 1))

        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        logger.info(
            "Epoch[{}/{}] Lr:{:.7f} Loss:{:.7f} L_MSE:{:.7f} L_rank:{:.7f} L_con:{:.7f} KT:{:.5f} MAPE:{:.5f} "
            "ErrBnd(0.01):{:.5f} Speed:{:.0f}/s Exa(h:m:s):{}".format(
                epoch_idx,
                config.epochs,
                lr,
                loss,
                loss_dict["loss_mse"],
                loss_dict["loss_rank"],
                loss_dict["loss_consist"],
                tau,
                acc,
                err,
                speed,
                exp_time,
            )
        )

        if (epoch_idx + 1) % config.test_freq == 0:
            acc, err, tau = infer(val_loader, net, config.dataset, config.device)
            if tau > best_tau:
                best_mape, best_error, best_tau = acc, err, tau
                save_check_point(
                    epoch_idx + 1,
                    batch_idx + 1,
                    config,
                    net.state_dict(),
                    None,
                    None,
                    False,
                    config.dataset + "_model_best.pth.tar",
                )

            if config.model_ema and config.model_ema_eval:
                acc_ema, err_ema, tau_ema = infer(
                    val_loader, model_ema.ema, config.dataset, config.device
                )
                if tau_ema > best_tau_ema:
                    best_mape_ema, best_error_ema, best_tau_ema = (
                        acc_ema,
                        err_ema,
                        tau_ema,
                    )
                    save_check_point(
                        epoch_idx + 1,
                        batch_idx + 1,
                        config,
                        get_state_dict(model_ema),
                        None,
                        None,
                        False,
                        config.dataset + "_model_best_ema.pth.tar",
                    )

            logger.info(
                "CheckPoint_TEST: KT {:.5f}, Best_KT {:.5f}, EMA_KT {:.5f}, Best_EMA_KT {:.5f} "
                "MAPE {:.5f}, Best_MAPE {:.5f}, EMA_MAPE {:.5f}, Best_EMA_MAPE {:.5f}, "
                "ErrBnd(0.01) {:.5f}, Best_ErrB {:.5f}, EMA_ErrBnd(0.01) {:.5f}, Best_EMA_ErrB {:.5f}, ".format(
                    tau,
                    best_tau,
                    tau_ema,
                    best_tau_ema,
                    acc,
                    best_mape,
                    acc_ema,
                    best_mape_ema,
                    err,
                    best_error,
                    err_ema,
                    best_error_ema,
                )
            )

        if (epoch_idx + 1) % config.save_epoch_freq == 0:
            logger.info("Saving Model after %d-th Epoch." % (epoch_idx + 1))
            save_check_point(
                epoch_idx + 1,
                batch_idx + 1,
                config,
                net.state_dict(),
                optimizer,
                scheduler,
                False,
                config.dataset + "_checkpoint_Epoch" + str(epoch_idx + 1) + ".pth.tar",
            )
        save_check_point(
            epoch_idx + 1,
            batch_idx + 1,
            config,
            net.state_dict(),
            optimizer,
            scheduler,
            False,
            config.dataset + "_latest.pth.tar",
        )
    logger.info(
        "Training Finished! Best MAPE: %11.8f, Best ErrBnd(0.01): %11.8f; Best MAPE on EMA: %11.8f, Best ErrBond(0.05) on EMA: %11.8f"
        % (best_mape, best_error, best_mape_ema, best_error_ema)
    )


@torch.no_grad()
def infer(dataloader, net, dataset, device=None, isTest=False):
    metric = Metric()
    net.eval()
    for bid, batch_data in enumerate(dataloader):
        if "nasbench" in dataset:
            gt = batch_data["test_acc_avg"] if isTest else batch_data["val_acc_avg"]
            if device != None:
                for k, v in batch_data.items():
                    batch_data[k] = v.to(device)
            logits = net(batch_data, None)
        elif dataset == "nnlqp":
            codes, gt, sf = (
                batch_data[0]["netcode"],
                batch_data[0]["cost"],
                batch_data[1],
            )
            logits = (
                net(None, None, codes.to(device), sf.to(device))
                if device != None
                else net(None, None, codes, None)
            )
        pre = (
            torch.cat([r.to(gt.device) for r in logits], dim=0)
            if isinstance(logits, list)
            else logits
        )
        ps = pre.data.cpu().numpy()[:, 0].tolist()
        gs = gt.data.cpu().numpy()[:, 0].tolist()
        metric.update(ps, gs)
        acc, err, tau = metric.get()
    return acc, err, tau


def test(config):
    test_loader = init_dataloader(config, logger)
    net = init_layers(config, logger)
    auto_load_model(config, net)
    if torch.cuda.is_available():
        net = net.cuda(config.device)
    acc, err, tau = infer(
        test_loader,
        net,
        config.dataset,
        config.device,
        isTest=True,
    )
    logger.info(
        f"Test with test acc: KT {tau:8.5f}, MAPE {acc:8.5f}, ErrBnd(0.01) {err:8.5f}"
    )
    acc, err, tau = infer(
        test_loader,
        net,
        config.dataset,
        config.device,
        isTest=False,
    )
    logger.info(
        f"Test with val acc: KT {tau:8.5f}, MAPE {acc:8.5f}, ErrBnd(0.01) {err:8.5f}"
    )


if __name__ == "__main__":
    args = argLoader()
    if os.path.exists(args.save_path):
        raise Exception("Dir already exit! Please check it!")
    else:
        os.makedirs(args.save_path)
    logname = "train" if args.do_train else "test"
    logname = os.path.join(args.save_path, f"{logname}.log")
    logger = setup_logger(logname)

    # setup_seed(args.seed)

    print("Totally", torch.cuda.device_count(), "GPUs are available.")
    if args.parallel:
        print("Using data parallel.")
        for device in range(torch.cuda.device_count()):
            print("Device: ", device, "Name: ", torch.cuda.get_device_name(device))
    else:
        torch.cuda.set_device(args.device)
        print(
            "Device: ", args.device, "Name: ", torch.cuda.get_device_name(args.device)
        )

    if args.do_train:
        logger.info("Configs: %s" % (args))
        train(args)
    else:
        test(args)
