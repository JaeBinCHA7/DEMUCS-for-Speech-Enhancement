from tensorboardX import SummaryWriter

class Writer(SummaryWriter):
    def __init__(self, logdir):
        super(Writer, self).__init__(logdir)

    def log_train_loss(self, loss_type, train_loss, step):
        self.add_scalar('train_{}_loss'.format(loss_type), train_loss, step)

    def log_valid_loss(self, loss_type, valid_loss, step):
        self.add_scalar('valid_{}_loss'.format(loss_type), valid_loss, step)

    def log_score(self, metrics_name, metrics, step):
        self.add_scalar(metrics_name, metrics, step)

    def log_wav(self, noisy_wav, clean_wav, enhanced_wav, step):
        # <Audio>
        self.add_audio('noisy_wav', noisy_wav, step, sample_rate=16000)
        self.add_audio('clean_target_wav', clean_wav, step, sample_rate=16000)
        self.add_audio('enhanced_wav', enhanced_wav, step, sample_rate=16000)
