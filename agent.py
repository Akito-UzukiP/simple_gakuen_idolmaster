import torch
import torch.nn as nn
import torch.nn.functional as F

class CardTransformer(nn.Module):
    def __init__(self, card_feature_dim, player_feature_dim, d_model=128, nhead=8, num_layers=2):
        super(CardTransformer, self).__init__()
        self.card_embedding = nn.Linear(card_feature_dim, d_model)
        self.player_embedding = nn.Linear(player_feature_dim, d_model)
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
        x_player = self.player_embedding(x_player)
        x_player = x_player.unsqueeze(1).repeat(1, x_cards.size(1), 1)
        x = x_cards + x_player
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x)
        x = x.permute(1, 0, 2)
        # mean pooling
        x = x.mean(dim=1)
        return x

# 测试模型
# model = CardTransformer(card_feature_dim=13, player_feature_dim=10,d_model=128)
# sample_cards = torch.randn(10, 5, 13)
# sample_player = torch.randn(10, 10)
# output = model(sample_cards, sample_player)
# print(output)
