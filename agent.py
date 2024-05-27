import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
class CardTransformer(nn.Module):
    def __init__(self,card_feature_dim, player_feature_dim, max_cards = 8, d_model=128, nhead=8, num_layers=2):
        super(CardTransformer, self).__init__()
        self.card_embedding = nn.Linear(card_feature_dim, d_model)
        self.player_embedding = nn.Linear(player_feature_dim, d_model)
        self.max_cards = max_cards
        # 自注意力
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers, batch_first=True)
        # 输出层
       # self.output_layer = nn.Linear(d_model, 1)

        # 初始化参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x_cards, x_player):
        x_cards = self.card_embedding(x_cards)
        # 将x_cards补全到max_cards，补全采用[-2] + [0] * (x_cards.size(-1) - 1)的方式
        if x_cards.size(1) < self.max_cards:
            padding = torch.zeros(x_cards.size(0), self.max_cards - x_cards.size(1), x_cards.size(-1)).to(x_cards.device)
            padding[:, :, 0] = -2
            x_cards = torch.cat([x_cards, padding], dim=1)
        x_player = self.player_embedding(x_player)
        x_player = x_player.unsqueeze(1).repeat(1, x_cards.size(1), 1)
        x = x_cards + x_player
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)
        x = x.mean(dim=1)
        return x
    
class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomExtractor, self).__init__(observation_space, features_dim)
        self.card_transformer = CardTransformer(card_feature_dim=19, player_feature_dim=13,d_model=features_dim)
    def forward(self, observations):
        game = observations['game']
        card = observations['card']
        card = self.card_transformer(card, game)
        return card