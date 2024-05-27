import effects
import random
import numpy as np
from cards import create_card
class Game:
    def __init__(self, hp = 30, total_turn = 6, target = 60):
        self.hp = hp
        self.init_hp = hp
        self.robust = 0
        self.good_impression = 0
        self.good_condition = 0
        self.best_condition = 0
        self.motivation = 0

        self.turn_left = total_turn
        self.init_turn = total_turn

        self.playable_cnt = 1

        self.score = 0
        self.target = target
        self.init_target = target
        self.hand = [] # 手牌
        self.deck = [] # 牌库
        self.discard = [] # 弃牌堆
        self.exile = [] # 除外

        self.effects = []

        self.is_over = False

        self.rest = create_card('休憩',cost=-2)

    def __str__(self) -> str:
        string = "Turns left: " + str(self.turn_left) + "\n"
        string += "Target: " + str(self.target) + "\n"
        string += "Playable count: " + str(self.playable_cnt) + "\n"
        string += 'HP: ' + str(self.hp) + ', '
        string += 'Robust: ' + str(self.robust) + ', ' if self.robust > 0 else ''
        string += 'Good Impression: ' + str(self.good_impression) + ', ' if self.good_impression > 0 else ''
        string += 'Good Condition: ' + str(self.good_condition) + ', ' if self.good_condition > 0 else ''
        string += 'Best Condition: ' + str(self.best_condition) + ', ' if self.best_condition > 0 else ''
        string += 'Motivation: ' + str(self.motivation) + ', ' if self.motivation > 0 else ''
        string += 'Score: ' + str(self.score) + '\n'
        string += 'Hand: ' + str(self.hand) + '\n'
        string += 'Deck: ' + str(self.deck) + '\n'
        string += 'Discard: ' + str(self.discard) + '\n'
        string += 'Exile: ' + str(self.exile)
        return string
    def __repr__(self) -> str:
        return self.__str__()

    def draw(self, num=1):
        ''' 抽牌，随机抽取牌库中的牌
        Input:
            num: 抽牌数量
        '''
        for i in range(num):
            if len(self.deck) == 0:
                self.deck = self.discard
                self.discard = []
                random.shuffle(self.deck)
            if len(self.deck) == 0:
                return
            self.hand.append(self.deck.pop())
            
    def check_playable(self, card_idx):
        ''' 检查是否可以打出某张牌
        Input:
            card_idx: 手牌中的牌的索引
        '''
        if card_idx >= len(self.hand):
            return False
        if self.playable_cnt == 0:
            return False
        if self.hand[card_idx].effects[0] > self.hp:
            return False
        if self.hand[card_idx].effects[1] > self.hp + self.robust:
            return False
        return True

    def play(self, card_idx):
        effects.effect_roll(self.hand[card_idx].effects, self)
        if self.hand[card_idx].name == '休憩':
            self.hand.pop(card_idx)
        else:
            self.discard_card(card_idx, True)
        self.playable_cnt -= 1

    def discard_card(self, card_idx, used = False):
        if self.hand[card_idx].exile and used:
            self.exile.append(self.hand.pop(card_idx))
        else:
            self.discard.append(self.hand.pop(card_idx))

    def start_round(self):
        self.draw(3)
        self.hand.append(self.rest)
        self.playable_cnt = 1

    def shuffle(self):
        self.deck = self.deck + self.discard + self.exile + self.hand
        self.discard = []
        self.exile = []
        self.hand = []
        random.shuffle(self.deck)

    def determine_is_over(self):
        if self.hp <= 0 or self.score >= self.target:
            self.is_over = True
        if self.turn_left == 0:
            self.is_over = True

    def end_round(self):
        if self.rest in self.hand:
            self.hand.remove(self.rest)
        for _ in range(len(self.hand)):
            self.discard_card(0)
        self.turn_left -= 1
        self.determine_is_over()
    def pseudo_end_round(self):
        # 无效操作，不消耗回合数
        pass
    def reset(self):
        self.hp = self.init_hp
        self.robust = 0
        self.good_impression = 0
        self.good_condition = 0
        self.best_condition = 0
        self.motivation = 0
        self.score = 0
        self.turn_left = self.init_turn
        self.is_over = False
        if self.rest in self.hand:
            self.hand.remove(self.rest)        
        self.shuffle()

    def observe(self):
        observation = []
        observation.append(self.hp)
        observation.append(self.robust)
        observation.append(self.good_impression)
        observation.append(self.good_condition)
        observation.append(self.best_condition)
        observation.append(self.motivation)
        observation.append(self.score)
        observation.append(self.target)
        observation.append(self.turn_left)
        observation.append(self.playable_cnt)

        card_observation = []
        for card in self.hand:
            if self.check_playable(self.hand.index(card)):
                tmp = [0]
            else:
                tmp = [-1]
            tmp.extend(card.observe())
            card_observation.append(tmp)
        for card in self.deck:
            tmp = [1]
            tmp.extend(card.observe())
            card_observation.append(tmp)
        for card in self.discard:
            tmp = [2]
            tmp.extend(card.observe())
            card_observation.append(tmp)
        for card in self.exile:
            tmp = [3]
            tmp.extend(card.observe())
            card_observation.append(tmp)
        observation = np.array(observation, dtype=np.float32)
        card_observation = np.array(card_observation, dtype=np.float32)
        return observation, card_observation