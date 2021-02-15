import torch
from torch import nn
import logging

from .conv import Conv2dTranspose, Conv2d

class Wav2LandRNN(nn.Module):
    def __init__(self):
        super(Wav2LandRNN, self).__init__()
        self.lstm = nn.LSTM(input_size=80, hidden_size=256, num_layers=3, batch_first=False,
                            bidirectional=False)
        self.mlp1 = nn.Linear(256 + 68 * 3, 512)
        self.mlp2 = nn.Linear(512, 256)
        self.mlp3 = nn.Linear(256, 68 * 3)

    def forward(self, x,src):
        x = x.permute(1, 0, 2)
        x, _ = self.lstm(x)
        x = x[-1, :, :]
        src_flat = torch.reshape(src, (src.shape[0], 68 * 3))
        x = torch.cat((x, src_flat), 1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = torch.reshape(x, (x.shape[0], 68, 3))
        return x

class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0), )

        self.mlp1 = nn.Linear(512 + 68 * 3, 512)
        self.mlp2 = nn.Linear(512, 256)
        self.mlp3 = nn.Linear(256, 68 * 3)

    def forward(self, audio_sequences,src):
        #logging.info("audio_sequences1: {}".format(audio_sequences.shape))
        audio_sequences = audio_sequences.permute(0,2,1)
        #logging.info("audio_sequences2: {}".format(audio_sequences.shape))
        audio_sequences = audio_sequences.unsqueeze(1)
        #logging.info("audio_sequences3: {}".format(audio_sequences.shape))
        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1
        #logging.info("audio_embedding1: {}".format(audio_embedding.shape))
        audio_embedding = torch.squeeze(audio_embedding)
        #logging.info("audio_embedding2: {}".format(audio_embedding.shape))
        if len(audio_embedding.shape)<2:
            audio_embedding = audio_embedding.unsqueeze(0)
        #logging.info("audio_embedding3: {}".format(audio_embedding.shape))
        src = torch.reshape(src, (src.shape[0],68*3))
        #logging.info("src: {}".format(src.shape))
        x = torch.cat((audio_embedding, src), 1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        x = torch.reshape(x, (-1, 68, 3))
        return x

def LoadWav2Lip(path, cpu=False):
    model = Wav2Lip()
    if cpu:
        model.cpu()
    else:
        model.cuda()

    checkpoint = torch.load(path, map_location="cpu" if cpu else "cuda")
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def LoadWav2LandRNN(path, cpu=False):
    model = Wav2LandRNN()
    if cpu:
        model.cpu()
    else:
        model.cuda()

    checkpoint = torch.load(path, map_location="cpu" if cpu else "cuda")
    model.load_state_dict(checkpoint)
    model.eval()
    return model

def LoadWav2LipMove(path, cpu=False):
    model = Wav2LipMove()
    if cpu:
        model.cpu()
    else:
        model.cuda()

    checkpoint = torch.load(path, map_location="cpu" if cpu else "cuda")
    model.load_state_dict(checkpoint)
    model.eval()
    return model

class Wav2LipMove(nn.Module):
    def __init__(self,window_size=18):
        super(Wav2LipMove, self).__init__()
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            #Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            #Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            #Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            #Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
        )
        self.window_size = window_size
        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=3, batch_first=False,
                            bidirectional=False)
        self.mlp = nn.Linear(256,4*4)

    def forward(self, audio_sequences):
        bs = audio_sequences.size(0)
        frames = audio_sequences.size(1)
        audio_sequences = torch.reshape(audio_sequences,
                                        (bs*frames,
                                         audio_sequences.size(2),audio_sequences.size(3)))
        audio_sequences = audio_sequences.permute(0, 2, 1)
        audio_sequences = audio_sequences.unsqueeze(1)
        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1
        audio_embedding = torch.squeeze(audio_embedding)
        audio_embedding = torch.reshape(audio_embedding,(bs,frames,audio_embedding.size(1)))
        audio_embedding = audio_embedding.permute(1,0,2)
        seq,_ = self.lstm(audio_embedding)
        seq = seq.permute(1, 0, 2)
        seq = self.mlp(seq)
        seq = torch.tanh(seq)
        seq = torch.reshape(seq, (bs,frames,4, 4))
        return seq
