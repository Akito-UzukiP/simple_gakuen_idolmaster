import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CardLayer(nn.Module):
    def __init__(self, card_info_dim, effect_info_dim, d_model, num_layers=2):
        '''
        对卡片的信息进行处理，使用GRU，对于每个卡片的信息，使用GRU进行处理，取最后一个hidden state作为卡片的表示
        '''
        super(CardLayer, self).__init__()
        self.effect_gru = nn.GRU(effect_info_dim, d_model, num_layers=1, batch_first=True)
        self.encoding_layer = nn.Linear(card_info_dim, d_model)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers, batch_first=True)

    def forward(self, cards_batch):
        '''
        cards_batch: [
            [
                {
                    'info': (card_info_dim),
                    'effect': (effects, effect_info_dim)
                },
                ...
            ],
            ...
        ]
        '''
        batch_size = len(cards_batch)
        card_counts = [len(cards) for cards in cards_batch]
        max_cards = max(card_counts)

        max_effects_per_card = 4
        effect_lengths = []  # 记录每个effect的实际长度
        for cards in cards_batch:
            card_effect_lengths = []
            for card in cards:
                #print(card['effect'])
                effect_len = len(card['effect'])
                max_effects_per_card = max(max_effects_per_card, effect_len)
                card_effect_lengths.append(effect_len)
            effect_lengths.append(card_effect_lengths)

        card_info_list = []
        effect_info_list = []

        for cards in cards_batch:
            card_info = torch.stack([torch.tensor(card['info'], dtype=torch.float32) for card in cards], dim=0) # (max_cards, card_info_dim)

            effects = [torch.tensor(card['effect'], dtype=torch.float32) for card in cards] 
            padded_effects = []
            for effect in effects:
                padded_effect = nn.functional.pad(effect, (0, 0, 0, max_effects_per_card - effect.size(0)))  # 在尾部填充
                padded_effects.append(padded_effect)

            effect_info = torch.stack(padded_effects, dim=0) # (max_cards, max_effects_per_card, effect_info_dim)

            # Pad card_info and effect_info to the maximum number of cards in the batch
            card_info = nn.functional.pad(card_info, (0, 0, 0, max_cards - len(cards)))
            effect_info = nn.functional.pad(effect_info, (0, 0, 0, 0, 0, max_cards - len(cards)))

            card_info_list.append(card_info)
            effect_info_list.append(effect_info)

        card_info = torch.stack(card_info_list, dim=0)  # (batch_size, max_cards, card_info_dim)
        effect_info = torch.stack(effect_info_list, dim=0)  # (batch_size, max_cards, max_effects_per_card, effect_info_dim)

        #print(card_info.size(), effect_info.size())
        # card_info: (batch_size, max_cards, card_info_dim)
        # effect_info: (batch_size, max_cards, max_effects_per_card, effect_info_dim)
        # 处理卡片信息
        card_info = self.encoding_layer(card_info)
        #print(card_info.size())
        # card_info: (batch_size, max_cards, d_model)
        # 处理效果信息
        batch_size, max_cards, max_effects_per_card, effect_info_dim = effect_info.size()
        effect_info = effect_info.view(batch_size * max_cards, max_effects_per_card, effect_info_dim)
        effect_info, _ = self.effect_gru(effect_info)

        # 取对应长度的hidden state
        effect_info = effect_info.view(batch_size, max_cards, max_effects_per_card, -1)
        effect_hidden_states = []
        for i, card_effect_lengths in enumerate(effect_lengths):
            batch_hidden_states = []
            for j, length in enumerate(card_effect_lengths):
                batch_hidden_states.append(effect_info[i, j, length - 1, :])  # 取对应长度的hidden state
            batch_hidden_states = torch.stack(batch_hidden_states, dim=0)
            padding_size = max_cards - batch_hidden_states.size(0)
            if padding_size > 0:
                batch_hidden_states = nn.functional.pad(batch_hidden_states, (0, 0, 0, padding_size))
            effect_hidden_states.append(batch_hidden_states)
        effect_hidden_states = torch.stack(effect_hidden_states, dim=0)
        #print(effect_hidden_states.size())
        # 组合卡片信息和效果信息
        x = card_info + effect_hidden_states

        # 通过GRU处理
        gru_output, _ = self.gru(x)

        return gru_output[:, -1, :]  # 取最后一个hidden state
    
class GameLayer(nn.Module):
    def __init__(self, player_info_dim, card_info_dim, effect_info_dim, d_model):
        '''
        对玩家的信息进行处理，使用MLP，
        '''
        super(GameLayer, self).__init__()

        self.player_embedding = nn.Linear(player_info_dim, d_model)
        self.card_layer = CardLayer(card_info_dim, effect_info_dim, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        ) 
        
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

        combined_embedding = player_embedding + card_embedding
        output = self.mlp(combined_embedding)
        
        return output

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, player_info_dim, card_info_dim, effect_info_dim, d_model):
        super(CustomExtractor, self).__init__(observation_space, d_model)
        self.game_layer = GameLayer(player_info_dim, card_info_dim, effect_info_dim, d_model)
        self.player_info_dim = player_info_dim
        self.card_info_dim = card_info_dim
        self.effect_info_dim = effect_info_dim

    def forward(self, observations):
        # # card部分展平，展平后的维度为(max_cards, card_info_dim + max_effects_per_card * effect_info_dim)，不足部分用0填充
        # card_observation = np.zeros((max_cards, card_info_dim + max_effects_per_card * effect_info_dim))
        # for i, card in enumerate(observation['card']):
        #     card_observation[i, :card_info_dim] = card['info']
        #     card_observation[i, card_info_dim:card['effect'].shape[0] * effect_info_dim] = card['effect'].flatten()
        # observation = {
        #     'game': observation['game'],
        #     'card': card_observation
        # }
        # return observation

        # 逆过程，将展平的card信息还原
        all_cards = []
        # b, c, d 
        b, c, d = observations['card'].shape
        for i in range(b):
            cards = []
            for j in observations['card'][i]:
                #print(j.shape)
                card_info = j[:self.card_info_dim]
                #print(card_info.shape)
                try:
                    effect_cnt = int(j[-1])
                except:
                    print(observations['card'].shape)
                    print(j.shape)
                    print(j[-1])
                    raise
                effect_info = j[self.card_info_dim:self.card_info_dim + effect_cnt * 49].reshape(effect_cnt, 49)
                cards.append({
                    'info': card_info,
                    'effect': effect_info
                })
            all_cards.append(cards)
        observations['card'] = all_cards

        return self.game_layer(observations)



if __name__ == "__main__":
    import numpy as np
    model = GameLayer(32, 15, 49, 128)
    cards_batch = []
    for _ in range(32):
        num_cards = np.random.randint(1, 5)
        cards = []
        for _ in range(num_cards):
            card_info = np.random.randn(15).astype(np.float32)
            effects = np.array([np.random.randn(49).astype(np.float32) for _ in range(np.random.randint(1, 5))])
            cards.append({
                'info': card_info,
                'effect': effects
            })
        cards_batch.append(cards)
    
    game_info = np.random.randn(32, 32).astype(np.float32)  # Batch size of 32
    x = {'game': torch.tensor(game_info), 'card': cards_batch}
    print(x)
    model_output = model(x)
    print(model_output.shape)