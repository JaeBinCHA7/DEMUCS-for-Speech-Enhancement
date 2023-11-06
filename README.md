# DEMUCS-for-Speech-Enhancement

Welcome to the DEMUCS-for-Speech-Enhancement repository.

DEMUCS is a source separation model proposed by Facebook (now META), which received great attention for its fast processing speed and excellent performance. It was later applied to the field of voice enhancement and showed excellent performance [1]. This repository provides the following research content:

1. Implementation of HD-DEMUCS[2] (successor of DEMUCS model)
2. DEMUCS in the time-frequency domain
3. HD-DEMUCS in time-frequency domain

Performance is provided at the end of the README, and as a result, you can check the performance comparison in HD-DEMUCS and the Time-frequency domain.

## Update
* **2023.11.06**

## Requirements 
This repo is implemented in Ubuntu 22.04, PyTorch 2.0.1, Python3.10, and CUDA11.7. For package dependencies, you can install them by:

```
pip install -r requirements.txt    
```

## Dataset Installation 
To get started with the DEMUCS-for-Speech-Enhancement project, the first step is to set up the dataset which will be used to train and evaluate the model. This project uses a combination of the Voice Bank corpus and DEMAND database 

**Voice Bank + DEMAND Dataset**: The dataset combines clean speech from the Voice Bank corpus and various types of noise from the DEMAND database to simulate realistic noisy speech conditions.

Download: https://datashare.ed.ac.uk/handle/10283/1942 

## Getting Started
1. Install the necessary libraries.   
2. Set directory paths for your dataset. ([options.py](https://github.com/JaeBinCHA7/DEMUCS-for-Speech-Enhancement/blob/main/options.py)) 
```   
# dataset path
noisy_dirs_for_train = '../Dataset/train/noisy/'   
noisy_dirs_for_valid = '../Dataset/valid/noisy/'   
```   
3. Run [train_interface.py](https://github.com/JaeBinCHA7/DEMUCS-for-Speech-Enhancement/blob/main/train_interface.py)

## Architecture
<center><img src = "https://github.com/JaeBinCHA7/DEMUCS-for-Speech-Enhancement/assets/87358781/6a427f68-4fe8-495f-995d-977388a4a1a5" width="100%" height="100%"></center>

## Results 
<center><img src = "https://github.com/JaeBinCHA7/DEMUCS-for-Speech-Enhancement/assets/87358781/6984f1c5-2e10-4254-8dc6-c4c4725eb902" width="100%" height="100%"></center>

## References   
**[1] Defossez, Alexandre, Gabriel Synnaeve, and Yossi Adi. "Real time speech enhancement in the waveform domain." arXiv preprint arXiv:2006.12847 (2020).** [[paper]](https://doi.org/10.48550/arXiv.2006.12847)  [[code]](https://github.com/facebookresearch/denoiser)   
**[2] Kim, Doyeon, et al. "HD-DEMUCS: General Speech Restoration with Heterogeneous Decoders." arXiv preprint arXiv:2306.01411 (2023).** [[paper]](https://doi.org/10.48550/arXiv.2306.01411)

## Contact  
E-mail: jbcha7@yonsei.ac.kr

