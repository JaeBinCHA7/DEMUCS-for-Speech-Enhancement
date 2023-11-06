"""
Test interface for speech enhancement!
You can just run this file.
"""
import argparse
import torch
import options
import utils
import random
import numpy as np
import time
from dataloader import create_dataloader
######################################################################################################################
#                                                  Parser init                                                       #
######################################################################################################################
opt = options.Options().init(argparse.ArgumentParser(description='speech enhancement')).parse_args()
print(opt)
######################################################################################################################
#                                                   Model init                                                       #
######################################################################################################################
# set device
DEVICE = torch.device(opt.device)
# set seeds
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
# define model
model = utils.get_arch(opt)
total_params = utils.cal_total_params(model)
print('total params   : %d (%.2f M, %.2f MBytes)\n' %
      (total_params,
       total_params / 1000000.0,
       total_params * 4.0 / 1000000.0))
# load the params
print('Load the pretrained model...')
chkpt = torch.load(opt.pretrain_model_path)
model.load_state_dict(chkpt['model'])
model = model.to(DEVICE)

######################################################################################################################
######################################################################################################################
#                                             Main program - train                                                   #
######################################################################################################################
######################################################################################################################

print('Test start...')
opt.test_database = opt.noisy_dirs_for_test
test_loader = create_dataloader(opt, mode='test')
data_num = 0
enh_all = []
cln_all = []
enh_all_total = []
cln_all_total = []
rtf = []

# test
# model = torch.compile(model)
model.eval()
with torch.no_grad():
    for inputs, targets in utils.Bar(test_loader):
        data_num += 1
        # to cuda
        inputs = inputs.float().to(DEVICE)
        targets = targets.float().to(DEVICE)

        st_time = time.time()
        # Predict targets
        outputs = model(inputs)
        outputs = outputs.squeeze(1)

        # Calculate inference time
        if_time = time.time() - st_time
        rtf.append(if_time / ((1 / opt.fs) * inputs.shape[1]))

        # get score
        enhanced_wavs = outputs.cpu().detach().numpy()
        clean_wavs = targets.cpu().detach().numpy()[:, :outputs.size(1)]

        # Add to total lists
        enh_all_total.extend(enhanced_wavs)
        cln_all_total.extend(clean_wavs)

        del inputs, targets, outputs
        torch.cuda.empty_cache()

# Calculate and print overall scores
avg_pesq_total = utils.cal_pesq_batch(enh_all_total, cln_all_total)
avg_stoi_total = utils.cal_stoi_batch(enh_all_total, cln_all_total)
avg_csig_total, avg_cbak_total, avg_covl_total, avg_ssnr_total = utils.cal_mos_batch(enh_all_total, cln_all_total)

print('\nOverall Scores:')
print('PESQ: {:.4f}  STOI: {:.4f}  CSIG {:.4f}  CBAK {:.4f}  COVL {:.4f}  SSNR {:.4f}'
      .format(avg_pesq_total, avg_stoi_total, avg_csig_total, avg_cbak_total, avg_covl_total, avg_ssnr_total))

print('RTF (Real-Time Factor) : {:.4f}'.format(np.mean(rtf)))
print('System has been finished.')
