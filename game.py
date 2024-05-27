import effects
import random
import numpy as np
from cards import create_card, Card, create_random_ktn_deck
class Game:
    def __init__(self, hp = 30, total_turn = 6, target = 60):
        self.hp = hp
        self.init_hp = hp
        # 元气
        self.robust = 0
        # 好印象
        self.good_impression = 0
        # 好调
        self.good_condition = 0
        # 绝好调
        self.best_condition = 0
        # 干劲
        self.motivation = 0

        # 体力消费增加
        self.hp_damage_increase = 0
        # 体力消费减少
        self.hp_damage_decrease = 0
        # 体力消费减少（直接）
        self.hp_damage_decrease_direct = 0

        # 额外抽牌
        self.additional_draw = 0

        self.turn_left = total_turn
        self.init_turn = total_turn
        self.current_turn = 0

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
            
    def deep_copy(self):
        ''' 深拷贝当前状态
        '''
        new_game = Game(self.hp, self.init_turn, self.target)
        new_game.hp = self.hp
        new_game.robust = self.robust
        new_game.good_impression = self.good_impression
        new_game.good_condition = self.good_condition
        new_game.best_condition = self.best_condition
        new_game.motivation = self.motivation
        new_game.score = self.score
        new_game.turn_left = self.turn_left
        new_game.current_turn = self.current_turn
        new_game.is_over = self.is_over
        new_game.playable_cnt = self.playable_cnt
        new_game.hand = self.hand.copy()
        new_game.deck = self.deck.copy()
        new_game.discard = self.discard.copy()
        new_game.exile = self.exile.copy()
        new_game.effects = self.effects.copy()
        return new_game

    def check_playable(self, card_idx):
        ''' 检查是否可以打出某张牌，思路：copy当前状态，尝试打出这张牌，确认所有值是否大于0
        Input:
            card_idx: 手牌中的牌的索引
        '''
        if card_idx >= len(self.hand):
            return False
        if self.playable_cnt == 0:
            return False
        new_game = self.deep_copy()
        new_game.play(card_idx)
        if new_game.hp < 0:
            return False
        if new_game.robust < 0:
            return False
        if new_game.good_impression < 0:
            return False
        if new_game.good_condition < 0:
            return False
        if new_game.best_condition < 0:
            return False
        if new_game.motivation < 0:
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
        # 处理第一回合必上手的卡
        cnt = 0
        determined_card = []
        if self.current_turn == 0:
            for i, card in enumerate(self.deck):
                if card.first:
                    cnt += 1
                    determined_card.append(i)
        #print(determined_card)
        for i in determined_card[::-1]:
            self.hand.append(self.deck.pop(i))
            
        self.draw(max(3-cnt, 0))
        self.hand.append(self.rest)
        self.playable_cnt = 1

    def shuffle(self):
        self.deck = self.deck + self.discard + self.exile + self.hand
        self.discard = []
        self.exile = []
        self.hand = []
        random.shuffle(self.deck)

    def determine_is_over(self):
        if self.hp < 0 or self.score >= self.target:
            self.is_over = True
        if self.turn_left == 0:
            self.is_over = True

    def end_round(self):
        if self.rest in self.hand:
            self.hand.remove(self.rest)
        for _ in range(len(self.hand)):
            self.discard_card(0)
        self.turn_left -= 1
        self.current_turn += 1
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
        self.current_turn = 0
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
        observation.append(self.hp_damage_increase)
        observation.append(self.hp_damage_decrease)
        observation.append(self.hp_damage_decrease_direct)
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


if __name__ == "__main__":
    game = Game()
    game.deck = create_random_ktn_deck()
    print(game.observe()[0].size, game.observe()[1].shape)