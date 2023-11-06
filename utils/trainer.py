import torch
from .progress import Bar
from .scores import cal_pesq_batch, cal_stoi_batch


######################################################################################################################
#                                               train loss function                                                  #
######################################################################################################################
def time_loss_train(model, train_loader, loss_calculator, optimizer, writer, EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)
        outputs = outputs.squeeze(1)
        loss = loss_calculator(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('total', train_loss, EPOCH)

    return train_loss


def mrstft_loss_train(model, train_loader, loss_calculator, optimizer, writer, EPOCH, DEVICE, opt):
    # initialization
    train_loss = 0
    batch_num = 0

    # train
    model.train()
    for inputs, targets in Bar(train_loader):
        batch_num += 1

        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        outputs = model(inputs)
        outputs = outputs.squeeze(1)

        sc_loss, mag_loss = loss_calculator[0](outputs, targets)
        mae_loss = loss_calculator[1](outputs, targets)
        loss = sc_loss + mag_loss + mae_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= batch_num

    # tensorboard
    writer.log_train_loss('total', train_loss, EPOCH)

    return train_loss


######################################################################################################################
#                                               valid loss function                                                  #
######################################################################################################################
def time_loss_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()
    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)
            outputs = outputs.squeeze(1)
            loss = loss_calculator(outputs, targets)

            valid_loss += loss

            # get score
            enhanced_wavs = outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = cal_pesq_batch(enhanced_wavs, clean_wavs)
            stoi = cal_stoi_batch(enhanced_wavs, clean_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('total', valid_loss, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi


def mrstft_loss_valid(model, valid_loader, loss_calculator, writer, EPOCH, DEVICE, opt):
    # initialization
    valid_loss = 0
    batch_num = 0
    avg_pesq = 0
    avg_stoi = 0

    # validation
    model.eval()
    with torch.no_grad():
        for inputs, targets in Bar(valid_loader):
            batch_num += 1

            # to cuda
            inputs = inputs.float().to(DEVICE)
            targets = targets.float().to(DEVICE)

            outputs = model(inputs)
            outputs = outputs.squeeze(1)

            sc_loss, mag_loss = loss_calculator[0](outputs, targets)
            mae_loss = loss_calculator[1](outputs, targets)
            loss = sc_loss + mag_loss + mae_loss

            valid_loss += loss

            # get score
            enhanced_wavs = outputs.cpu().detach().numpy()
            clean_wavs = targets.cpu().detach().numpy()

            pesq = cal_pesq_batch(enhanced_wavs, clean_wavs)
            stoi = cal_stoi_batch(enhanced_wavs, clean_wavs)

            avg_pesq += pesq
            avg_stoi += stoi

        valid_loss /= batch_num
        avg_pesq /= batch_num
        avg_stoi /= batch_num

    # tensorboard
    writer.log_valid_loss('total', valid_loss, EPOCH)
    writer.log_score('PESQ', avg_pesq, EPOCH)
    writer.log_score('STOI', avg_stoi, EPOCH)
    writer.log_wav(inputs[0], targets[0], outputs[0], EPOCH)

    return valid_loss, avg_pesq, avg_stoi