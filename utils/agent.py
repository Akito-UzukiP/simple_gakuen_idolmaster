import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class CardLayer(nn.Module):
    def __init__(self, card_info_dim, effect_info_dim, d_model, num_layers=2, max_effects_per_card=4, max_cards=5):
        '''
        '''
        super(CardLayer, self).__init__()
        self.encoding_layer = nn.Linear(card_info_dim, d_model)
        self.effect_encoding_layer = nn.Linear(effect_info_dim, d_model,bias=False)
        self.hidden_layer = nn.Linear(d_model, d_model)
        self.card_info_dim = card_info_dim
        self.effect_info_dim = effect_info_dim
        self.d_model = d_model
        self.max_effects_per_card = max_effects_per_card
        self.max_cards = max_cards


    def forward(self, cards_batch):
        '''
        cards_batch-> (batch_size, max_cards, card_info_dim + max_effects_per_card * effect_info_dim)
        '''
        b, c, _ = cards_batch.shape
        card_info = cards_batch[:, :, :self.card_info_dim] # -> (batch_size, max_cards, card_info_dim)
        effect_info = cards_batch[:, :, self.card_info_dim:].reshape(-1, self.max_effects_per_card, self.effect_info_dim) # -> (batch_size * max_cards, max_effects_per_card, effect_info_dim)
        card_info = self.encoding_layer(card_info) # -> (batch_size, max_cards, d_model)
        effect_info = self.effect_encoding_layer(effect_info) # -> (batch_size * max_cards, max_effects_per_card, d_model)
        effect_info = effect_info.sum(dim=1) # -> (batch_size * max_cards, d_model)
        effect_info = effect_info.reshape(b, c, -1) # -> (batch_size, max_cards, d_model)
        card_info = card_info + effect_info
        card_info = self.hidden_layer(card_info)
        card_info = torch.relu(card_info)
        return card_info
    
class GameLayer(nn.Module):
    def __init__(self, player_info_dim, card_info_dim, effect_info_dim, d_model, max_cards=5, max_effects_per_card=4):
        '''
        对玩家的信息进行处理，使用MLP，
        '''
        super(GameLayer, self).__init__()

        self.player_embedding = nn.Linear(player_info_dim, d_model)
        self.card_layer = CardLayer(card_info_dim, effect_info_dim, d_model, max_effects_per_card=max_effects_per_card, max_cards=max_cards)
        self.max_cards = max_cards
        self.mlp = nn.Sequential(
            nn.Linear(d_model * self.max_cards, d_model),
            nn.ReLU(),
        ) 
        self.d_model = d_model
        
    def forward(self, x):
        '''
        x: {
            'game': (batch_size, player_info_dim),
            'card': [
                [
                    {
                        'info': (card_info_dim),
                        'effect': (effects, effect_info_dim)
                    },
                    ...
                ],
                ...
            ]
        }
        '''
        player_info = x['game']
        cards_batch = x['card']

        player_embedding = self.player_embedding(player_info)
        card_embedding = self.card_layer(cards_batch)
        #print(card_embedding.shape)
        card_embedding = card_embedding.reshape(-1, self.max_cards * self.d_model)
        card_embedding = self.mlp(card_embedding)
        #print(card_embedding.shape, player_embedding.shape)
        output = card_embedding + player_embedding
        
        return output

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, player_info_dim, card_info_dim, effect_info_dim, d_model, max_cards=5, max_effects_per_card=4):
        super(CustomExtractor, self).__init__(observation_space, d_model)
        self.game_layer = GameLayer(player_info_dim, card_info_dim, effect_info_dim, d_model, max_cards=max_cards, max_effects_per_card=max_effects_per_card)
        self.player_info_dim = player_info_dim
        self.card_info_dim = card_info_dim
        self.effect_info_dim = effect_info_dim

    def forward(self, observations):
        # observation = {
        #     'game': observation['game'], -> (batch_size, player_info_dim)
        #     'card': card_observation -> (batch_size, max_cards, card_info_dim + max_effects_per_card * effect_info_dim)
        # }
        # return observation
        return self.game_layer(observations)
    






if __name__ == "__main__":
    import numpy as np
    model = GameLayer(32, 15, 49, 128, 5, 4)
    print(model)
    cards_batch = []
    for _ in range(256):
        num_cards = 5
        cards = []
        for _ in range(num_cards):
            card_info = np.random.randn(15).astype(np.float32)
            effects = np.array([np.random.randn(49).astype(np.float32) for _ in range(4)])
            cards.append(np.concatenate([card_info, effects.flatten()]))
        cards_batch.append(cards)
    cards_batch = torch.tensor(cards_batch).float()
    print(cards_batch.shape)
    game_info = np.random.randn(256, 32).astype(np.float32)  # Batch size of 32
    x = {'game': torch.tensor(game_info), 'card': cards_batch}
    #print(x)
    model_output = model(x)
    print(model_output.shape)