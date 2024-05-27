import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
class CardTransformer(nn.Module):
    def __init__(self,card_feature_dim, player_feature_dim, max_cards = 30, d_model=128, nhead=8, num_layers=2):
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
# class CustomActorCritic(nn.Module):
#     def __init__(self, observation_space, action_space, card_feature_dim, player_feature_dim, d_model=128, nhead=8, num_layers=2):
#         super(CustomActorCritic, self).__init__()
#         self.card_transformer = CardTransformer(card_feature_dim, player_feature_dim, d_model, nhead, num_layers)
#         self.action_head = nn.Linear(d_model, 1)
#         self.value_head = nn.Linear(d_model, 1)

#     def forward(self, x_cards, x_player):
#         features = self.card_transformer(x_cards, x_player)
#         # 对于每一个卡片，输出一个动作概率
#         action_logits = self.action_head(features)
#         action_logits = action_logits.squeeze(-1)
#         # mean_pooling
#         features = features.mean(dim=1)
#         values = self.value_head(features)
#         return action_logits, values

#     def act(self, x_cards, x_player):
#         action_logits, values = self.forward(x_cards, x_player)
#         action_probs = F.softmax(action_logits, dim=-1)
#         dist = Categorical(action_probs)
#         action = dist.sample()
#         return action, dist.log_prob(action), dist.entropy(), values

#     def get_value(self, x_cards, x_player):
#         _, values = self.forward(x_cards, x_player)
#         return values

#     def evaluate_actions(self, x_cards, x_player, action):
#         action_logits, values = self.forward(x_cards, x_player)
#         action_probs = F.softmax(action_logits, dim=-1)
#         dist = Categorical(action_probs)
#         return dist.log_prob(action), dist.entropy(), values


# class Agent:
#     def __init__(self, envs):
#         self.envs = envs
#         self.actor_critic = CustomActorCritic(
#             envs.single_observation_space,
#             envs.single_action_space,
#             card_feature_dim=13,
#             player_feature_dim=10,
#             d_model=128,
#             nhead=8,
#             num_layers=2
#         )

#     def get_action_and_value(self, obs, action=None):
#         x_cards = obs['card']
#         x_player = obs['game']
#         if action is None:
#             return self.actor_critic.act(x_cards, x_player)
#         else:
#             return self.actor_critic.evaluate_actions(x_cards, x_player, action)
#     def to(self, device):
#         self.actor_critic.to(device)
#         return self
# 测试模型
# model = CardTransformer(card_feature_dim=13, player_feature_dim=10,d_model=128)
# sample_cards = torch.randn(10, 5, 13)
# sample_player = torch.randn(10, 10)
# output = model(sample_cards, sample_player)
# print(output)
