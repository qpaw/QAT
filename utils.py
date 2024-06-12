import copy
import datetime
import os
import random
import time
from collections import defaultdict, deque
from random import randrange

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.v2
from torch import linalg as LA
from torchvision.transforms.functional import InterpolationMode

import qrelu
import utils


class ClassStrategyScore:
    def __init__(self, batch, targets, device="cpu") -> None:
        self._batch = batch
        self._targets = targets
        self._device = device
        self.dim = batch.shape[0]

    def set_forward_hooks(self, model=None):
        """
        set hoooks for activations: ReLU and FinalQRelu
        """

        def hook_fn(module, input, output):  # def hook fn
            output[output != 0] = 1
            view_output.append(output)

        if not model:
            model = self

        for child_name, child in model.named_children():
            if isinstance(child, nn.ReLU) or isinstance(
                child, qrelu.FinalQRelu
            ):
                # child.register_forward_hook(hook_fn)  # set hook
                child.register_full_backward_hook(hook_fn)
            else:
                self.set_hooks(child)

    def set_hooks(self, model=None):
        """
        set hoooks for activations: ReLU and FinalQRelu
        """

        def hook_fn(module, grad_input, grad_output):  # def hook fn
            # module, grad_input, grad_output
            grad_out = copy.deepcopy(grad_output[0])
            # grad_out[grad_out != 0.0] = 1
            grad_out[grad_out > 0.01] = 1
            grad_out[grad_out <= 0.01] = 0
            view_output.append(grad_out)

        if not model:
            model = self

        for child_name, child in model.named_children():
            if (
                isinstance(child, nn.ReLU)
                or isinstance(child, qrelu.FinalQRelu)
                or isinstance(child, qrelu.FinalQSigmoid)
            ):
                # child.register_forward_hook(hook_fn)  # set hook
                child.register_full_backward_hook(hook_fn)
            else:
                self.set_hooks(child)

    def get_matrix(self, model):
        """
        get activation matrix by train batch
        use hooks
        size of result: list of activations len activations count
        shape of each activation depend on nets structure

        if activation > 0, then return 1
        if activation == 0, thrn retrun 0
        """
        net = model
        global view_output  # for save hooks results
        view_output = []
        self.set_hooks(net)

        net.to(self._device)
        net.train(True)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            net.parameters(),
            # lr=0.06,
            lr=0.01,
        )
        outputs = net(self._batch)
        loss = criterion(outputs, self._targets)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        net.eval()
        # TODO: hook.remove()
        return view_output

    def get_hamming(self, matrix):
        A = 0  # len of activations
        for line, act_map in enumerate(matrix):
            act_map = act_map.view(act_map.shape[0], -1)  # reshape by one dim
            A += act_map.shape[1]
            if line == 0:
                h_matrix = torch.cdist(act_map, act_map, p=0)
            else:
                h_matrix += torch.cdist(act_map, act_map, p=0)

        h_matrix = A - h_matrix
        score = torch.log(LA.matrix_norm(h_matrix))  # Frobenius default
        return h_matrix, score.item()

    def get_score(self, model):
        matrix = self.get_matrix(model)
        # return matrix
        score = self.get_hamming(matrix)
        return score


class ClassStratageGenerator:
    def __init__(
        self, bit=32, active_func=qrelu.FinalQRelu, active_deriv=qrelu.dReLU
    ) -> None:
        self._bit = bit
        self._active_func = active_func
        self._active_deriv = active_deriv
        pass

    def get_activations_count(self, model):
        return model.get_activations_count()

    def apply_strategy(self, model, act_bit_list):
        """
        3, 4, 6, 32 - how many bit to use
        0 - is no change
        """
        model.change_by_qactivations(bitlist=act_bit_list)

    def _uniform_generator(self, layers, budget, max_bit=None):
        # if max_bit is not None:
        #     b_max = max_bit
        # else:
        #     b_max = self._bit
        # b_max = 0
        raise ValueError("Uniform strategy is not implemented jet")

    def _random_generator(self, layers, budget, max_bit=None):
        if max_bit is not None:
            b_max = max_bit
        else:
            b_max = self._bit
        result_list = []
        saved_list = []

        for _ in range(layers):
            b = randrange(1, b_max)
            if sum(saved_list) < budget:
                result_list.append(b)
                saved_list.append(self._bit - b)

        # add zeros if need
        if len(result_list) < layers:
            diff = layers - len(result_list)
            for _ in range(diff):
                result_list.append(0)

        random.shuffle(result_list)
        return result_list

    def generate(
        self, model, percent_reduce=15, strat_type="random", bit_max=None
    ):
        """
        Generate new startegy
        """

        act_count = self.get_activations_count(model)
        budget = int(round((act_count * percent_reduce) / 100)) * self._bit
        if strat_type == "random":
            lst = self._random_generator(act_count, budget, bit_max)
            model.change_by_qactivations(
                bitlist=lst,
                active_func=self._active_func,
                active_deriv=self._active_deriv,
            )
            return model, lst
        if strat_type == "uniform":
            pass  # TODO: future relise


class ClassificationPreset:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
        use_v2=False,
    ):
        T = torchvision.transforms.v2
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(
                f"backend can be 'tensor' or 'pil', but got {backend}"
            )

        transforms += [
            T.Resize(
                (resize_size, resize_size),
                interpolation=interpolation,
                antialias=True,
            ),
            # T.CenterCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms += [
            (
                T.ToDtype(torch.float, scale=True)
                if use_v2
                else T.ConvertImageDtype(torch.float)
            ),
            T.Normalize(mean=mean, std=std),
        ]

        if use_v2:
            transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class SmoothedValue:
    """
    getted from here
    https://github.com/pytorch/vision/blob/main/references/classification/

    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger:
    """
    getted from here
    https://github.com/pytorch/vision/blob/main/references/classification

    Log all metricks avare training.
    """

    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str}")


def train_one_epoch(
    model,
    criterion,
    optimizer,
    data_loader,
    device,
    epoch,
    args,
    model_ema=None,
    scaler=None,
    run=None,
):
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter(
        "img/s", SmoothedValue(window_size=10, fmt="{value}")
    )

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(
        metric_logger.log_every(data_loader, args.print_freq, header)
    ):
        start_time = time.time()
        image, target = image.to(device), target.to(device)

        output = model(image)
        loss = criterion(output, target)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(
            loss=loss.item(), lr=optimizer.param_groups[0]["lr"]
        )
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(
            batch_size / (time.time() - start_time)
        )

    if run:
        run.log(
            {
                "acc1": metric_logger.acc1.global_avg,
                "epoch": epoch,
            }
        )

        run.log(
            {
                "acc5": metric_logger.acc5.global_avg,
                "epoch": epoch,
            }
        )

        run.log(
            {
                "loss": metric_logger.loss.global_avg,
                "epoch": epoch,
            }
        )

        # run.log(
        #     {
        #         "acc1": metric_logger.acc1.global_avg,
        #         "acc5": acc5.item(),
        #         "loss": loss.item(),
        #         "img/s": batch_size / (time.time() - start_time),
        #     }
        # )

        # run.log(
        #     {
        #         "acc1": metric_logger.acc1.global_avg,
        #         "acc5": acc5.item(),
        #         "loss": loss.item(),
        #         "img/s": batch_size / (time.time() - start_time),
        #     }
        # )

        # run.log(
        #     {
        #         "acc1": acc1.item()})
        # run.log(
        #     {
        #         "acc5": acc5.item()})
        # run.log(
        #         {
        #         "loss": loss.item()})
        # run.log(
        #     {
        #         "img/s": batch_size / (time.time() - start_time)})


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top
    predictions for the specified values of k
    """
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)
        if target.ndim == 2:
            target = target.max(dim=1)[1]

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def evaluate(
    model, criterion, data_loader, device, print_freq=100, log_suffix=""
):
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in metric_logger.log_every(
            data_loader, print_freq, header
        ):
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    print(
        f"{header} Acc@1 {metric_logger.acc1.global_avg:.3f}\
            Acc@5 {metric_logger.acc5.global_avg:.3f}"
    )
    # return metric_logger.acc1.global_avg


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join(
        "~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt"
    )
    cache_path = os.path.expanduser(cache_path)
    return cache_path


def load_data(traindir, valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size, train_crop_size = (
        args.val_resize_size,
        None,  # args.val_crop_size,
        None,  # args.train_crop_size,
    )
    interpolation = InterpolationMode("bilinear")

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_train from {cache_path}")
        dataset, _ = torch.load(cache_path, weights_only=False)
    else:
        # auto_augment_policy = getattr(args, "auto_augment", None)
        # random_erase_prob = getattr(args, "random_erase", 0.0)
        # ra_magnitude = getattr(args, "ra_magnitude", None)
        # augmix_severity = getattr(args, "augmix_severity", None)
        dataset = torchvision.datasets.ImageFolder(
            traindir,
            ClassificationPreset(
                crop_size=train_crop_size,
                interpolation=interpolation,
                resize_size=val_resize_size,
                # auto_augment_policy=auto_augment_policy,
                # random_erase_prob=random_erase_prob,
                # ra_magnitude=ra_magnitude,
                # augmix_severity=augmix_severity,
                # backend=args.backend,
                # use_v2=args.use_v2,
            ),
        )
        if args.cache_dataset:
            print(f"Saving dataset_train to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            # utils.save_on_master((dataset, traindir), cache_path)
            torch.save((dataset, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if args.cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print(f"Loading dataset_test from {cache_path}")
        dataset_test, _ = torch.load(cache_path, weights_only=False)
    else:
        # if args.weights and args.test_only:
        #     weights = torchvision.models.get_weight(args.weights)
        #     preprocessing = weights.transforms(antialias=True)
        #     if args.backend == "tensor":
        #         preprocessing = torchvision.transforms.Compose(
        #             [torchvision.transforms.PILToTensor(), preprocessing]
        #         )

        # else:
        preprocessing = ClassificationPreset(
            crop_size=val_crop_size,
            resize_size=val_resize_size,
            interpolation=interpolation,
            # backend=args.backend,
            # use_v2=args.use_v2,
        )

        dataset_test = torchvision.datasets.ImageFolder(
            valdir,
            preprocessing,
        )
        if args.cache_dataset:
            print(f"Saving dataset_test to {cache_path}")
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((dataset_test, valdir), cache_path)

    print("Creating data loaders")

    train_sampler = torch.utils.data.SequentialSampler(
        dataset
    )  # RandomSampler TODO
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, dataset_test, train_sampler, test_sampler
