import time
import torch

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


# calculate the size of total network
def cal_total_params(our_model):
    total_parameters = 0
    for variable in our_model.parameters():
        shape = variable.size()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim
        total_parameters += variable_parameters

    return total_parameters

def cal_params_per_module(model):
    params_per_module = {}

    for name, module in model.named_children():
        total_params = sum(p.numel() for p in module.parameters())
        params_per_module[name] = total_params

    return params_per_module

# def cal_total_params(our_model):
#     return utils.parameters_to_vector(our_model.parameters()).numel()

class Bar(object):
    def __init__(self, dataloader):
        if not hasattr(dataloader, 'dataset'):
            raise ValueError('Attribute `dataset` not exists in dataloder.')
        if not hasattr(dataloader, 'batch_size'):
            raise ValueError('Attribute `batch_size` not exists in dataloder.')

        self.dataloader = dataloader
        self.iterator = iter(dataloader)
        self.dataset = dataloader.dataset
        self.batch_size = dataloader.batch_size
        self._idx = 0
        self._batch_idx = 0
        self._time = []
        self._DISPLAY_LENGTH = 50

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._time) < 2:
            self._time.append(time.time())

        self._batch_idx += self.batch_size
        if self._batch_idx > len(self.dataset):
            self._batch_idx = len(self.dataset)

        try:
            batch = next(self.iterator)
            self._display()
        except StopIteration:
            raise StopIteration()

        self._idx += 1
        if self._idx >= len(self.dataloader):
            self._reset()

        return batch

    def _display(self):
        if len(self._time) > 1:
            t = (self._time[-1] - self._time[-2])
            eta = t * (len(self.dataloader) - self._idx)
        else:
            eta = 0

        rate = self._idx / len(self.dataloader)
        len_bar = int(rate * self._DISPLAY_LENGTH)
        bar = ('=' * len_bar + '>').ljust(self._DISPLAY_LENGTH, '.')
        idx = str(self._batch_idx).rjust(len(str(len(self.dataset))), ' ')

        tmpl = '\r{}/{}: [{}] - ETA {:.1f}s'.format(
            idx,
            len(self.dataset),
            bar,
            eta
        )
        print(tmpl, end='')
        if self._batch_idx == len(self.dataset):
            print()

    def _reset(self):
        self._idx = 0
        self._batch_idx = 0
        self._time = []


def complex_cat(inputs, dim=1):
    real, imag = [], []
    for idx, data in enumerate(inputs):
        r, i = torch.chunk(data, 2, dim)
        real.append(r)
        imag.append(i)
    real = torch.cat(real, dim)
    imag = torch.cat(imag, dim)
    outputs = torch.cat([real, imag], dim=1)
    return outputs


def power_compress(x, cut_len=257):
    real = x[:, :cut_len]
    imag = x[:, cut_len:]
    mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-7)
    phase = torch.atan2(imag, real)
    mags = mags ** 0.3 + 1e-7
    real_compress = mags * torch.cos(phase)
    imag_compress = mags * torch.sin(phase)
    return real_compress, imag_compress


def power_uncompress(real, imag):
    mags = torch.sqrt(real ** 2 + imag ** 2 + 1e-7)
    phase = torch.atan2(imag, real)
    mags = mags ** (1. / 0.3) + 1e-7
    real_compress = mags * torch.cos(phase)
    imag_compress = mags * torch.sin(phase)
    return torch.cat([real_compress, imag_compress], 1)