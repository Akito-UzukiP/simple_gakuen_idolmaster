import gym
from gym import spaces
import numpy as np
from game import Game
import cards
import random
class GakuenIdolMasterEnv(gym.Env):
    def __init__(self):
        super(GakuenIdolMasterEnv, self).__init__()
        
        self.game = Game(hp=30, total_turn=8, target=60)
        self.game.deck = cards.create_ktn_deck()
        self.game.shuffle()
        self.game.start_round()
        
        # 动作空间：选择手牌中的一张卡
        self.max_cards = 8
        self.action_space = spaces.Discrete(self.max_cards)
        #print(self.max_cards)
        # 观察空间
        self.env_shape = self.game.observe()[0].shape[0]
        self.card_shape = (self.max_cards, 19)
        #print(self.env_shape, self.card_shape)
        self.observation_space = spaces.Dict({
            'game': spaces.Box(low=-10, high=10, shape=(self.env_shape,), dtype=np.float32),
            'card': spaces.Box(low=-10, high=10, shape=self.card_shape, dtype=np.float32)
        })
        self.current_score = 0
        self.seed()
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    def _get_obs(self):
        observation = self.game.observe()
        # observation[1]要补到30，用[-1] + [0]*18
        card_observation = np.array(observation[1])
        num_rows_to_add = self.max_cards - observation[1].shape[0]
        fill_value = np.array([[-1] + [0]*18] * num_rows_to_add)
        card_observation = np.vstack((observation[1], fill_value))
        return {
            'game': np.array(observation[0]),
            'card': card_observation
        }
    def reset(self):
        hp = random.randint(10,30)
        total_turn = random.randint(5,11)
        target = random.randint(total_turn*10, total_turn*20)
        self.game = Game(hp=hp, total_turn=total_turn, target=target)
        self.game.deck = cards.create_random_ktn_deck()
        #print(self.game.observe())
        self.game.start_round()
        self.current_score = 0
        return self._get_obs()
    def step(self, action):
        # 无效动作
        #print(action)
        if not self.game.check_playable(action):
            self.game.pseudo_end_round()
            return self._get_obs(), -20, False, {}
        self.game.play(action)
        self.game.end_round()
        done = self.game.is_over
        reward = self.game.score - self.current_score
        if done:
            reward += self.game.hp *1
            reward += self.game.turn_left * 10
        self.current_score = self.game.score
        self.game.start_round()
        return self._get_obs(), reward, done, {}
    
    def render(self, mode='human'):
        if mode == 'human':
            print(self.game)
# env = GakuenIdolMasterEnv()
# env._get_obs()