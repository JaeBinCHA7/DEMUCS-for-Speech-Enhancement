# get architecture
def get_arch(opt):
    arch = opt.arch

    print('You choose ' + arch + '...')
    if arch == 'DEMUCS':
        from models import DEMUCS
        model = DEMUCS()
    elif arch == 'DEMUCS_TF':
        from models import DEMUCS_TF
        model = DEMUCS_TF()
    elif arch == 'HDDEMUCS':
        from models import HDDEMUCS
        model = HDDEMUCS()
    elif arch == 'HDDEMUCS_TF':
        from models import HDDEMUCS_TF
        model = HDDEMUCS_TF()
    else:
        raise Exception("Arch error!")

    return model


# get trainer and validator (train method)
def get_train_mode(opt):
    loss_type = opt.loss_type

    print('You choose ' + loss_type + '...')
    if loss_type == 'time':
        from .trainer import time_loss_train
        from .trainer import time_loss_valid
        trainer = time_loss_train
        validator = time_loss_valid
    elif loss_type == 'mrstft':  # multiple(joint) loss function
        from .trainer import mrstft_loss_train
        from .trainer import mrstft_loss_valid
        trainer = mrstft_loss_train
        validator = mrstft_loss_valid
    else:
        raise Exception("Loss type error!")

    return trainer, validator


def get_loss(opt):
    from torch.nn import L1Loss
    from torch.nn.functional import mse_loss
    loss_oper = opt.loss_oper

    print('You choose loss operation with ' + loss_oper + '...')
    if loss_oper == 'l1':
        loss_calculator = L1Loss()
    elif loss_oper == 'l2':
        loss_calculator = mse_loss
    elif loss_oper == 'mrstft':
        from .loss import MultiResolutionSTFTLoss
        # loss_calculator = MultiResolutionSTFTLoss().to(opt.device)
        loss_fn_1 = MultiResolutionSTFTLoss().to(opt.device)
        loss_fn_2 = L1Loss()
        loss_calculator = [loss_fn_1, loss_fn_2]
    else:
        raise Exception("Arch error!")

    return loss_calculator
