import gym
from gym import spaces
import numpy as np
from game import Game
import cards

class GakuenIdolMasterEnv(gym.Env):
    def __init__(self):
        super(GakuenIdolMasterEnv, self).__init__()
        
        self.game = Game(hp=30, total_turn=8, target=60)
        self.game.deck = cards.create_ktn_deck()
        self.game.shuffle()
        self.game.start_round()
        
        # 动作空间：选择手牌中的一张卡
        self.max_cards = self.game.observe()[1].shape[0]
        self.action_space = spaces.Discrete(self.max_cards)
        print(self.max_cards)
        # 观察空间
        self.env_shape = self.game.observe()[0].shape[0]
        self.card_shape = self.game.observe()[1].shape
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
        return {
            'game': self.game.observe()[0],
            'card': self.game.observe()[1]
        }
    def reset(self):
        self.game.reset()
        #print(self.game.observe())
        self.game.start_round()
        self.current_score = 0
        return self._get_obs()
    def get_action_mask(self):
        mask = np.zeros(self.max_cards)
        # 对于所有[0]位为0的卡，可以被打出
        for i in range(self.max_cards):
            if self.game.observe()[1][i][0] == 0:
                mask[i] = 1
        return mask
    def step(self, action):
        # 无效动作
        if not self.game.check_playable(action):
            self.game.pseudo_end_round()
            return self._get_obs(), -10, False, {}
        self.game.play(action)
        self.game.end_round()
        done = self.game.is_over
        reward = self.game.score - self.current_score
        if done:
            reward += self.game.hp *3
        self.current_score = self.game.score
        self.game.start_round()
        return self._get_obs(), reward, done, {}
    def render(self, mode='human'):
        print(self.game)
env = GakuenIdolMasterEnv()
env.reset()

