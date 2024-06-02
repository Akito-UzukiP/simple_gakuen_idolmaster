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
max_cards = 8  # 假设每个玩家最多有10张卡片
max_effects_per_card = 4  # 假设每张卡片最多有5个效果

# 定义卡片效果空间
effect_space = spaces.Box(low=-np.inf, high=np.inf, shape=(max_effects_per_card, effect_info_dim), dtype=np.float32)

# 定义卡片信息空间
card_space = spaces.Dict({
    'info': spaces.Box(low=-np.inf, high=np.inf, shape=(card_info_dim,), dtype=np.float32),
    'effect': effect_space  # 使用固定最大效果数量来表示
})

# 定义观察空间




class GakuenIdolMasterEnv(gym.Env):
    def __init__(self, max_cards=5, max_hand_cards=5):
        super(GakuenIdolMasterEnv, self).__init__()
        
        self.max_stamina = 27
        self.total_turn = 11
        self.target = 100
        self.additional_cards = random.randint(5, 15)
        self.plus_cards = random.randint(3, 8)
        self.game = Game(max_stamina=self.max_stamina, turn_left=self.total_turn, target_lesson=self.target)
        self.deck = cards_future.random_hiro_deck(self.additional_cards, self.plus_cards)
        self.game.deck = self.deck
        self.game.shuffle_all_to_deck()
        # 动作空间：选择手牌中的一张卡
        self.max_cards = max_cards
        self.max_hand_cards = max_hand_cards

        self.action_space = spaces.Discrete(self.max_hand_cards, 0)
        observation_space = spaces.Dict({
            'game': spaces.Box(low=-np.inf, high=np.inf, shape=(player_info_dim,), dtype=np.float32),
            'card': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_cards, card_info_dim + max_effects_per_card * effect_info_dim), dtype=np.float32)  # 固定最大卡片数量和效果数量
        })
        self.observation_space = observation_space
        #print(self.max_cards)
        # 观察空间

        

        self.current_lesson = 0
        self.seed()

        self.last_steps = [-2] * self.game.turn_left


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
        self.total_turn = random.randint(6, 12)
        self.target = random.randint(self.total_turn*10, self.total_turn*20)
        self.game = Game(max_stamina=self.max_stamina, turn_left=self.total_turn, target_lesson=self.target)
        self.additional_cards = random.randint(5, 15)
        self.plus_cards = random.randint(3, 8)
        self.deck = cards_future.random_hiro_deck(self.additional_cards, self.plus_cards)
        self.game.deck = self.deck
        self.game.shuffle_all_to_deck()
        #print(self.game.observe())
        self.game.start_turn()
        self.current_lesson = 0
        return self._get_obs()
    def step(self, action):
        # 无效动作
        #print(action)
        reward = 0

        # 重复动作惩罚
        cnt = 0
        for i in range(len(self.last_steps)-1, -1, -1):
            if action == self.last_steps[i]:
                cnt +=1
                if cnt >= 4:
                    reward -= 30
                    #print("重复动作")
                    return self._get_obs(), -50 + self.game.current_turn*5 + reward, True, {}
            else :
                break
        self.last_steps.pop(0)
        self.last_steps.append(int(action))
        if action == self.max_hand_cards-1:
            self.game.rest()
        else:
            if not self.game.check_playable(action):
                #print("无效动作")
                # 找到第一个可打出的牌
                return self._get_obs(), -50 + self.game.current_turn*5 + reward, True, {}
            self.game.play_card(action)
        if self.game.playable_value == 0:
            self.game.end_turn()
        done = self.game.is_over
        self.game.lesson = min(self.game.lesson, self.game.target_lesson)
        reward += self.game.lesson - self.current_lesson
        if done:
            if self.game.lesson >= self.game.target_lesson:
                reward += 200
                reward += self.game.stamina * 8
                reward += self.game.turn_left * 100
            reward += self.game.review * 0.5
            reward += self.game.block * 0.7
                #reward += self.game.best_condition * 20
            reward += len(set(self.last_steps)) * 5
        self.current_lesson = self.game.lesson
        if self.game.playable_value == 0:
            self.game.start_turn()
        #print(reward)
        #print(reward)
        return self._get_obs(), reward, done, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            print(self.game)

if __name__ == "__main__":
    env = GakuenIdolMasterEnv()
    print(env.step(0))
