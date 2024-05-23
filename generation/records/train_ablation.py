from train import train
class Config:
    def __init__(self):
        self.embedding_dim=512
        self.device='cuda:3'
        self.hidden_size=512
        self.num_layers=6
        self.batch_size=64
        self.lr=2e-4
        self.weight_decay=2e-3
        self.num_epoch=10
        self.save_interval=1
        self.save_dir='models_ablation'
        self.seq2seq=True
        self.model_type = 'lstm'
        self.start_epoch = 0
        self.load = None
args = Config()
args.num_epoch = 20
args.load = None
args.start_epoch = 0
train(args)