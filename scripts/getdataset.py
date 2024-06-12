# import torch
# from torch.utils.data import Dataset
# from torchvision import datasets
# from torchvision.transforms import ToTensor
import os


def main(args):
    print(args.only_val)

    datasetpath = os.path.join(args.path, args.dataset)
    if os.path.exists(datasetpath):
        print(
            "Dir '{}' is exist. Remove old data and continue".format(
                datasetpath
            )
        )
        exit(0)
    os.makedirs(datasetpath)

    if args.dataset == "ImageNet":
        print("Start load {}...".format(args.dataset))
        if not os.path.exists(
            os.path.join(args.path, "ILSVRC2012_devkit_t12.tar.gz")
        ):
            import wget

            print("dowloading devkit \n")
            wget.download(
                "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",  # noqa: E501
                out=datasetpath,
            )
            if not args.only_val:
                print("dowloading train \n")
                wget.download(
                    "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",  # noqa: E501
                    out=datasetpath,
                )
            print("dowloading val \n")
            wget.download(
                "https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",  # noqa: E501
                out=datasetpath,
            )

        print(
            "Download finished. Next, you must extract data use script 'extract_ILSVRC.sh' or manual"  # noqa: E501
        )

    else:
        print("exit")


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(
        description="Load in you workspace dataset", add_help=add_help
    )

    parser.add_argument(
        "-d",
        "--dataset",
        default="imagenet",
        type=str,
        help="Load datasets from torch",
        choices=["ImageNet", "cifar", "mnist"],
    )

    parser.add_argument(
        "-p",
        "--path",
        default="./datasets",
        type=str,
        help="Where dataset load to",
    )

    parser.add_argument(
        "-v",
        "--only-val",
        dest="only_val",
        action="store_true",
        help="load only validation datset near 6.3 Gb",
    )
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
