import gym
from gym import spaces
import numpy as np
import random
try:
    from . import game_future, cards_future, effects_future, triggers_future
    from .game_future import Game
except:
    import game_future, cards_future, effects_future, triggers_future
    from game_future import Game

player_info_dim = 32
card_info_dim = 15
effect_info_dim = 49
max_cards = 40  # 假设每个玩家最多有10张卡片
max_effects_per_card = 4  # 假设每张卡片最多有5个效果

# 定义卡片效果空间
effect_space = spaces.Box(low=-np.inf, high=np.inf, shape=(max_effects_per_card, effect_info_dim), dtype=np.float32)

# 定义卡片信息空间
card_space = spaces.Dict({
    'info': spaces.Box(low=-np.inf, high=np.inf, shape=(card_info_dim,), dtype=np.float32),
    'effect': effect_space  # 使用固定最大效果数量来表示
})

# 定义观察空间
observation_space = spaces.Dict({
    'game': spaces.Box(low=-np.inf, high=np.inf, shape=(player_info_dim,), dtype=np.float32),
    'card': spaces.Box(low=-np.inf, high=np.inf, shape=(max_cards, card_info_dim + max_effects_per_card * effect_info_dim), dtype=np.float32)  # 固定最大卡片数量和效果数量
})




class GakuenIdolMasterEnv(gym.Env):
    def __init__(self):
        super(GakuenIdolMasterEnv, self).__init__()
        
        self.game = Game(max_stamina=100, turn_left=80, target_lesson=600)
        self.game.deck = cards_future.all_logic_cards
        self.game.shuffle_all_to_deck()
        self.game.deck = self.game.deck[:random.randint(30, 40)]
        # 动作空间：选择手牌中的一张卡
        self.max_cards = 35
        self.max_hand_cards = 8

        self.action_space = spaces.Discrete(self.max_hand_cards)
        self.observation_space = observation_space
        #print(self.max_cards)
        # 观察空间

        

        self.current_lesson = 0
        self.seed()

        self.game.start_turn()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    def _get_obs(self):
        observation = self.game.observe()
        # obs = {
        #     "game": np.array
        #     "card":[{
        #         "info": np.array
        #         "effect": np.array
        #     } for i in deck_info]
        # }
        # card部分展平，展平后的维度为(max_cards, card_info_dim + max_effects_per_card * effect_info_dim)，不足部分用0填充
        card_observation = np.zeros((max_cards, card_info_dim + max_effects_per_card * effect_info_dim))
        for i, card in enumerate(observation['card']):
            card_observation[i, :card_info_dim] = card['info']
            #print(card['effect'].shape[0] * effect_info_dim)
            #print(card['effect'].flatten().shape)
            card_observation[i, card_info_dim:card_info_dim+card['effect'].shape[0] * effect_info_dim] = card['effect'].flatten()
        observation = {
            'game': observation['game'],
            'card': card_observation
        }
        return observation
    def reset(self):
        max_stamina = 30
        total_turn = 8
        target = 60
        self.game = Game(max_stamina=100, turn_left=80, target_lesson=600)
        self.game.deck = cards_future.all_logic_cards
        self.game.shuffle_all_to_deck()
        self.game.deck = self.game.deck[:random.randint(30, 40)]
        #print(self.game.observe())
        self.game.start_turn()
        self.current_lesson = 0
        return self._get_obs()
    def step(self, action):
        # 无效动作
        #print(action)
        reward = 0

        action = action + 1
        if action == self.max_hand_cards:
            self.game.rest()
        else:
            if not self.game.check_playable(action):
                #print("无效动作")
                # 找到第一个可打出的牌
                return self._get_obs(), -50 + self.game.current_turn*5, True, {}
            self.game.play_card(action)
        if self.game.playable_value == 0:
            self.game.end_turn()
        done = self.game.is_over
        self.game.lesson = min(self.game.lesson, self.game.target_lesson)
        reward += self.game.lesson - self.current_lesson
        if done:
            if self.game.lesson >= self.game.target_lesson:
                reward += 500
                reward += self.game.stamina * 50
                reward += self.game.turn_left * 100
            reward += self.game.review * 50
            reward += self.game.block * 3
                #reward += self.game.best_condition * 20
        self.current_lesson = self.game.lesson
        if self.game.playable_value == 0:
            self.game.start_turn()
        return self._get_obs(), reward, done, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            print(self.game)

if __name__ == "__main__":
    env = GakuenIdolMasterEnv()
    print(env.step(0))
