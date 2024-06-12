import datetime
import os

# import argparse
import random
import time

import torch
import torchvision
from torch import nn
from torch.utils.data.dataloader import default_collate

import qrelu
import utils
from resnet50 import ResNet50

# from torchvision.models import resnet50  # , ResNet50_Weights

# import torch.nn as nn
# import torch.optim as optim


def main(args):
    print(args)
    tags = ["Final", "ResNet50", "QAT"]

    # Setting seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # np.random.seed(seed)
    random.seed(seed)

    print("Creating model")

    if args.model == "custom":
        path_model = args.checkpoints_custom
        tags.append("custom")
        tags.append(path_model)
        checkpointmodel = torch.load(path_model)
        model = ResNet50(num_classes=args.num_classes)
        model.load_state_dict(checkpointmodel["model"])

    if args.model == "manual":
        print("[INFO]: Training the manual ResNet50 model...")
        tags.append("manual")
        model = ResNet50(num_classes=args.num_classes)
    if args.model == "torch":
        print("[INFO]: Training the Torchvision ResNet50 model...")
        model = torchvision.models.resnet50()
        tags.append("torch")
    if args.model == "quantized":
        print(
            "[INFO]: Training the manual ResNet50 model \
                  with quantized activation func..."
        )
        if args.quant_activation == "qRelu":
            model = ResNet50(
                num_classes=args.num_classes,
                activ_func="FinalQReLU",
                bits=args.quant_bit,
                arange=tuple(map(int, args.quant_range.split(", "))),
            )
            tags.append("qRelu")
            tags.append(str(args.quant_bit))

    if args.model == "quantizedBest":
        print(
            "[INFO]: Training the manual ResNet50 model \
                  with quantized activation func \
                  by best strategy..."
        )
        base_model = ResNet50()

        train_dir = os.path.join(args.data_path, "val")  # TODO: return train
        val_dir = os.path.join(args.data_path, "val")
        dataset_calibrate, dataset_test, train_sampler, test_sampler = (
            utils.load_data(train_dir, val_dir, args)
        )

        data_loader_calibrate = torch.utils.data.DataLoader(
            dataset_calibrate,
            batch_size=args.batch_size,
            sampler=train_sampler,
            num_workers=args.workers,
            # pin_memory=True,
            collate_fn=default_collate,
        )
        x, y = next(iter(data_loader_calibrate))
        sscor = utils.ClassStrategyScore(x, y)
        # _, A_score = sscor.get_score(base_model)
        # print("best original score {}".format())

        if args.quant_activation == "qRelu":
            sgen = utils.ClassStratageGenerator(
                bit=args.quant_bit,
                active_func=qrelu.FinalQRelu,
                active_deriv=qrelu.dReLU,
            )
            tags.append("qRelu")
            tags.append(str(args.quant_bit))

        if args.quant_activation == "qSigmoid":
            sgen = utils.ClassStratageGenerator(
                bit=args.quant_bit,
                active_func=qrelu.FinalQSigmoid,
                active_deriv=qrelu.dSigmoid,
            )
            tags.append("qSigmoid")
            tags.append(str(args.quant_bit))

        for _ in range(args.count_examples):
            generatedModel, list_activations = sgen.generate(
                model=base_model,
                percent_reduce=args.quant_percent_reduse,
                bit_max=args.quant_bit,
                strat_type="random",
            )

            _, A_score = sscor.get_score(generatedModel)
            checkpoinGen = {
                "model": generatedModel.state_dict(),
                # "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                # "epoch": epoch,
                # "args": args,
            }
            torch.save(
                checkpoinGen,
                os.path.join(
                    args.checkpoints_dir,
                    f"genmodel_{args.quant_percent_reduse}reduce_{args.quant_bit}bit_{A_score}score.pth",
                ),
            )
        print("All complite succes")
        return 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if args.statistic:
        statistic = dict()
        statistic["models"] = args.model
        # Total parameters and trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        statistic["total parameters"] = total_params
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        statistic["training parameters"] = total_trainable_params
        statistic["model"] = model
        print(statistic)

    if args.data_path:
        train_dir = os.path.join(args.data_path, "val")  # TODO: return train
        val_dir = os.path.join(args.data_path, "val")
        dataset_train, dataset_test, train_sampler, test_sampler = (
            utils.load_data(train_dir, val_dir, args)
        )
    else:
        # dataset, dataset_test, train_sampler, test_sampler
        # = load_data(train_dir, val_dir, args)
        pass

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        # pin_memory=True,
        collate_fn=default_collate,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        sampler=test_sampler,
        num_workers=args.workers,
        # pin_memory=True,
        collate_fn=default_collate,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    opt_name = args.opt
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            # momentum=args.momentum,
            # weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            model.parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            eps=0.0316,
            alpha=0.9,
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters, lr=args.lr, weight_decay=args.weight_decay
        )

    print("Start training...")
    if args.log_wandb:
        import wandb

        assert wandb.run is None

        run = wandb.init(
            project="Quantization aware training",
            notes="Experiments with ResNet and quantization by binarization",
            tags=tags,
        )

        assert wandb.run is not None

        wandb.config = {
            "model_type": args.model,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "optimizer": opt_name,
        }
    else:
        run = None

    start_time = time.time()
    for epoch in range(0, args.epochs):
        utils.train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            data_loader=data_loader_train,
            device=device,
            epoch=epoch,
            args=args,
            run=run,
        )
        utils.evaluate(model, criterion, data_loader_test, device=device)
        if args.checkpoints_dir and args.checkpoints_dir != ".":
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                # "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch,
                "args": args,
            }
            torch.save(
                checkpoint,
                os.path.join(
                    args.output_dir, f"model_{args.model}_{epoch}.pth"
                ),
            )
            torch.save(
                checkpoint,
                os.path.join(args.output_dir, "checkpoint_{args.model}.pth"),
            )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    if args.log_wandb:
        wandb.finish()


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Work to test new quantization strategy", add_help=add_help
    )

    parser.add_argument(
        "-m",
        "--model",
        default="manual",
        type=str,
        help="What the model use: standart torch, manual, quantized",
        choices=["torch", "manual", "quantized", "quantizedBest", "custom"],
    )

    parser.add_argument(
        "-c",
        "--count-examples",
        default=10,
        type=int,
        help="How much examples will be generated",
    )

    parser.add_argument(
        "--checkpoints-custom",
        type=str,
        help="Path to load checkpoints for model",
    )

    parser.add_argument(
        "-qa",
        "--quant-activation",
        default="qRelu",
        type=str,
        help="Type quantized activation: qRelu or qSigmoid",
        choices=["qRelu", "qSigmoid"],
    )

    parser.add_argument(
        "-qb",
        "--quant-bit",
        default=8,
        type=int,
        help="Size of quantization",
    )

    parser.add_argument(
        "-qpr",
        "--quant-percent-reduse",
        default=50,
        type=int,
        help="Percent of quantization reduse",
    )

    parser.add_argument(
        "-qr",
        "--quant-range",
        default="-1, 1",
        type=str,
        help="Range of quantization",
    )

    parser.add_argument(
        "--opt",
        default="sgd",
        type=str,
        help="optimizer. Only four: SGD, SGD with Nesterov, RMSprop or AdamW",
        choices=["sgd", "sgd_nesterov", "rmsprop", "adamw"],
    )

    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")

    parser.add_argument(
        "--label-smoothing",
        default=0.0,
        type=float,
        help="label smoothing (default: 0.0)",
        dest="label_smoothing",
    )
    parser.add_argument(
        "--epochs",
        default=20,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "-b", "--batch-size", default=8, type=int, help="total batch size"
    )

    parser.add_argument(
        "-j",
        "--workers",
        default=1,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="device (Use cuda or cpu Default: cuda)",
    )
    parser.add_argument(
        "--num_classes", default=1000, type=int, help="number of classes"
    )
    parser.add_argument(
        "--val_resize_size",
        default=256,
        type=int,
        help="Size image for validation. Equal for test",
    )

    parser.add_argument(
        "--data-path",
        default="./datasets/ImageNet",
        type=str,
        help="dataset path or name for existing",
    )

    parser.add_argument(
        "--statistic",
        help="return list with statistic and train paramrters",
        action="store_true",
    )

    parser.add_argument(
        "--checkpoints-dir",
        default=".",
        type=str,
        help="path to save checkpoints",
    )

    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization.\
              It also serializes the transforms",
        action="store_true",
    )

    parser.add_argument(
        "--print-freq", default=100, type=int, help="print frequency"
    )

    parser.add_argument(
        "--log-vandb",
        dest="log_wandb",
        help="Begin loggin with wandb. \
            The <ID> must be locate in root dir in file 'wan.db'",
        action="store_true",
    )

    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
