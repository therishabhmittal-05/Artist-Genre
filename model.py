import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

class ArtGen(nn.Module):
    def __init__(self, n_artist, n_genre):
        super(ArtGen, self).__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.dropout = nn.Dropout(0.3)
        self.fc_artist = nn.Linear(base.fc.in_features, n_artist)
        self.fc_genre = nn.Linear(base.fc.in_features, n_genre)
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        artist_op = self.fc_artist(x)
        genre_op = self.fc_genre(x)

        return artist_op, genre_op