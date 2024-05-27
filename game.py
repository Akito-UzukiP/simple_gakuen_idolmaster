import effects
import random
class Game:
    def __init__(self, hp = 30, total_turn = 6, target = 60):
        self.hp = hp
        self.robust = 0
        self.good_impression = 0
        self.good_condition = 0
        self.best_condition = 0
        self.motivation = 0

        self.turn_left = total_turn

        self.playable_cnt = 1

        self.score = 0
        self.target = target
        self.hand = [] # 手牌
        self.deck = [] # 牌库
        self.discard = [] # 弃牌堆
        self.exile = [] # 除外

        self.effects = []

        self.is_over = False

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
            

    def play(self, card_idx):
        effects.effect_roll(self.hand[card_idx].effects, self)
        self.discard_card(card_idx, True)
        self.playable_cnt -= 1
    def skip(self):
        self.hp +=2
        effects.effect_roll([0, 0, 0, 0, 0, 0, 0, 0, 0], self)
    
    def discard_card(self, card_idx, used = False):
        if self.hand[card_idx].exile and used:
            self.exile.append(self.hand.pop(card_idx))
        else:
            self.discard.append(self.hand.pop(card_idx))

    def start_round(self):
        self.draw(3)
        self.playable_cnt = 1

    def shuffle(self):
        self.deck = self.deck + self.discard + self.exile
        self.discard = []
        self.exile = []
        random.shuffle(self.deck)

    def determine_is_over(self):
        if self.hp <= 0 or self.score >= self.target:
            self.is_over = True
        if self.turn_left == 0:
            self.is_over = True

    def end_round(self):
        for _ in range(len(self.hand)):
            self.discard_card(0)
        self.turn_left -= 1
        self.determine_is_over()