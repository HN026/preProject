import torch
from models import cifar_vgg


def get_model(model_type, dataset, plan=None):

    if dataset == 'cifar10':
        input_shape, num_classes = (3, 32, 32), 10
    elif dataset == 'cifar100':
        input_shape, num_classes = (3, 32, 32), 100

    # Available model list

    cifar100_models = {
        'vgg11': cifar_vgg.vgg11_bn,
        'vgg19': cifar_vgg.vgg19_bn,
        'vgg19-rwd-cl1': cifar_vgg.vgg19_rwd_cl1,
        'vgg19-rwd-cl2': cifar_vgg.vgg19_rwd_cl2,
        'vgg19-rwd-st36': cifar_vgg.vgg19_rwd_st36,
        'vgg19-rwd-st59': cifar_vgg.vgg19_rwd_st59,
        'vgg19-rwd-st79': cifar_vgg.vgg19_rwd_st79,
        'vgg19dbl': cifar_vgg.vgg19dbl,
        'vgg19dbl-rwd-st36': cifar_vgg.vgg19dbl_rwd_st36,
        'vgg19dbl-rwd-st59': cifar_vgg.vgg19dbl_rwd_st59,
        'vgg19dbl-rwd-st79': cifar_vgg.vgg19dbl_rwd_st79,
        'vgg-custom': cifar_vgg.vgg_custom
    }

    models = {
        'cifar100': cifar100_models,
    }

    # Checks whether the input model-dataset is supported.
    if dataset not in models:
        raise ValueError(f"{dataset} is not supported.")
    if model_type not in models[dataset]:
        raise ValueError(f"{model_type} is not supported in {dataset}. Check the dataset or model_type.\n"
                         f"Supported model in {dataset}:\n"
                         f"{list(models[dataset].keys())}")

    if "custom" in model_type:
        model = models[dataset][model_type](input_shape, num_classes, plan)
    else:
        model = models[dataset][model_type](input_shape, num_classes)
    model.model_type = model_type

    return model
