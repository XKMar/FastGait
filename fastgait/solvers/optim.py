# Modified from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/optim/optimizer.py  # noqa

import warnings

import torch
import torch.nn as nn

AVAI_OPTIMS = ["adam", "adamw", "amsgrad", "sgd", "rmsprop"]


def build_optimizer(
    models,
    optim="adam",
    lr=0.00035,
    weight_decay=5e-4,
    momentum=0.9,
    sgd_dampening=0,
    sgd_nesterov=False,
    rmsprop_alpha=0.99,
    adam_beta1=0.9,
    adam_beta2=0.999,
    staged_lr=False,
    new_layers="",
    base_lr_mult=0.1,
):
    """A function wrapper for building an optimizer.
    Args:
        models (List[nn.Module]): models.
        optim (str, optional): optimizer. Default is "adam".
        lr (float, optional): learning rate. Default is 0.0003.
        weight_decay (float, optional): weight decay (L2 penalty). Default is 5e-04.
        momentum (float, optional): momentum factor in sgd. Default is 0.9
        sgd_dampening (float, optional): dampening for momentum. Default is 0.
        sgd_nesterov (bool, optional): enables Nesterov momentum. Default is False.
        rmsprop_alpha (float, optional): smoothing constant for rmsprop. Default is 0.99
        adam_beta1 (float, optional): beta-1 value in adam. Default is 0.9.
        adam_beta2 (float, optional): beta-2 value in adam. Default is 0.99,
        staged_lr (bool, optional): uses different learning rates for base and new
            layers. Base layers are pretrained layers while new layers are randomly
            initialized, e.g. the identity classification layer. Enabling ``staged_lr``
            can allow the base layers to be trained with a smaller learning rate
            determined by ``base_lr_mult``, while the new layers will take the ``lr``.
            Default is False.
        new_layers (str or list): attribute names in ``model``. Default is empty.
        base_lr_mult (float, optional): learning rate multiplier for base layers.
            Default is 0.1.
    Examples::
        >>> # A normal optimizer can be built by
        >>> optimizer = torchreid.optim.build_optimizer(model, optim='sgd', lr=0.01)
        >>> # If you want to use a smaller learning rate for pretrained layers
        >>> # and the attribute name for the randomly initialized layer is 'classifier',
        >>> # you can do
        >>> optimizer = torchreid.optim.build_optimizer(
        >>>     model, optim='sgd', lr=0.01, staged_lr=True,
        >>>     new_layers='classifier', base_lr_mult=0.1
        >>> )
        >>> # Now the `classifier` has learning rate 0.01 but the base layers
        >>> # have learning rate 0.01 * 0.1.
        >>> # new_layers can also take multiple attribute names. Say the new layers
        >>> # are 'fc' and 'classifier', you can do
        >>> optimizer = torchreid.optim.build_optimizer(
        >>>     model, optim='sgd', lr=0.01, staged_lr=True,
        >>>     new_layers=['fc', 'classifier'], base_lr_mult=0.1
        >>> )
    """

    if optim not in AVAI_OPTIMS:
        raise ValueError(
            "Unsupported optim: {}. Must be one of {}".format(optim, AVAI_OPTIMS)
        )

    if isinstance(models, nn.Module):
        model_list = [models]
    else:
        model_list = models

    param_groups = []
    for model in model_list:

        if not isinstance(model, nn.Module):
            raise TypeError(
                "model given to build_optimizer must be an instance of nn.Module"
            )

        if staged_lr:
            if isinstance(new_layers, str):
                if new_layers is None:
                    warnings.warn(
                        "new_layers is empty, therefore, staged_lr is useless"
                    )
                new_layers = [new_layers]

            if isinstance(model, nn.parallel.DistributedDataParallel):
                model = model.module

            base_params = []
            base_layers = []
            new_params = []

            for name, module in model.named_children():
                if name in new_layers:
                    new_params += [p for p in module.parameters() if p.requires_grad]
                else:
                    base_params += [p for p in module.parameters() if p.requires_grad]
                    base_layers.append(name)

            params = [
                {"params": base_params},
                {"params": new_params, "lr": 0., "weight_decay": 0.,}
            ]

        else:
            params = [
                {"params": [value]}
                for value in model.parameters()
                if value.requires_grad
            ]

        param_groups.extend(params)

    # build optimizer
    if optim == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    
    elif optim == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=weight_decay,
            amsgrad=False,
        )

    elif optim == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optim == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optim == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    return optimizer
