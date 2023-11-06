"""
Docstring for Options
"""


class Options:
    def __init__(self):
        pass

    def init(self, parser):
        # global settings
        parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        parser.add_argument('--nepoch', type=int, default=60, help='training epochs')
        parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=5e-4, help='initial learning rate')
        parser.add_argument("--decay_epoch", type=int, default=30, help="epoch from which to start lr decay")
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--finetuned', type=bool, default=False, help='weight initialized')

        # train settings
        # parser.add_argument('--arch', type=str, default='DEMUCS', help='architecture')
        parser.add_argument('--arch', type=str, default='HDDEMUCS', help='architecture')
        # parser.add_argument('--arch', type=str, default='DEMUCS_TF', help='architecture')
        # parser.add_argument('--arch', type=str, default='HDDEMUCS_TF', help='architecture')

        parser.add_argument('--loss_type', type=str, default='mrstft', help='loss function type')
        parser.add_argument('--loss_oper', type=str, default='mrstft', help='loss function operation type')
        parser.add_argument('--c'
                            '', type=list, default=[0.1, 0.9, 0.05, 0.05], help='coupling constant')
        parser.add_argument('--device', type=str, default='cuda', help='gpu or cpu')

        # network settings

        # pretrained
        parser.add_argument('--env', type=str, default='231106', help='log name')
        parser.add_argument('--pretrained', type=bool, default=False, help='load pretrained_weights')
        parser.add_argument('--pretrain_model_path', type=str, default='./log/HDDEMUCS_231106/models/chkpt_50.pt',
                            help='path of pretrained_weights')
        # dataset
        parser.add_argument('--database', type=str, default='VBD', help='database')
        parser.add_argument('--fft_len', type=int, default=512, help='fft length')
        parser.add_argument('--win_len', type=int, default=400, help='window length')
        parser.add_argument('--hop_len', type=int, default=100, help='hop length')
        parser.add_argument('--fs', type=int, default=16000, help='sampling frequency')
        parser.add_argument('--chunk_size', type=int, default=32000, help='chunk size')

        parser.add_argument('--noisy_dirs_for_train', type=str,
                            default='../dataset/VBD/train/noisy/',
                            help='noisy dataset addr for train')
        parser.add_argument('--noisy_dirs_for_valid', type=str,
                            default='../dataset/VBD/test/noisy/',
                            help='noisy dataset addr for valid')
        parser.add_argument('--noisy_dirs_for_test', type=str,
                            default='../dataset/VBD/test/noisy/',
                            help='noisy dataset addr for test')

        return parser
