import random
import numpy as np
import math
class Game:
    '''
    Game类,用于描述游戏状态
    包含属性：
    stamina: 体力
    block: 元気/元气
    card_play_aggressive: やる気/干劲
    lesson: 分数
    review: 好印象
    lesson_buff: 集中
    parameter_buff: 好调
    parameter_buff_multiple_per_turn: 绝好调

    playable_value: 可打出牌数

    stamina_consumption_add: 体力消费增加
    stamina_consumption_down: 体力消费减少
    stamina_consumption_down_fix: 体力消费减少（直接）
    timer_card_draw: 额外抽牌, 包含下一回合、下下回合的额外抽牌数, list[2],当多张牌累计时, 分别加入
    timer_lesson: 后续回合额外分数，目前只有下一回合
    timer_card_upgrade: 后续回合额外升级牌数，可能有下回合、下下回合，可能累计
    anti_debuff: 抗debuff 次数
    block_restriction: 元气增长无效
    search_effect_play_count_buff: 下一张卡效果起效2次
    exam_status_enchant: 各种累计触发效果


    timer结算规则:
    各timer在触发后立刻结算并减少timer
    好调、绝好调、好印象
    当好调、（绝好调？）从0变为非0时，当回合不消耗好调
    例如：

    '''
    def __init__(self):
        self.max_stamina = 30
        self.stamina = 30
        self.block = 0
        self.card_play_aggressive = 0
        self.lesson = 0
        self.review = 0
        self.lesson_buff = 0
        self.parameter_buff = 0
        self.parameter_buff_multiple_per_turn = 0

        self.card_draw = 3
        self.playable_value = 1

        self.stamina_consumption_add = 0
        self.stamina_consumption_down = 0
        self.stamina_consumption_down_fix = 0

        self.timer_card_draw = [0, 0]
        self.timer_lesson = 0
        self.timer_card_upgrade = [0, 0]
        self.anti_debuff = 0
        self.block_restriction = 0
        self.search_effect_play_count_buff = 0
        self.exam_status_enchant = [0] * 8

        # 手牌、牌库、弃牌堆、除外
        self.hand = []
        self.deck = []
        self.discard = []
        self.exile = []

        # 回合数、目标分数
        self.turn_left = 6
        self.target = 60
        self.current_turn = 0

    def draw(self, num):
        '''
        抽牌
        '''
        for _ in range(num):
            if len(self.deck) == 0:
                self.deck = self.discard
                self.discard = []
                random.shuffle(self.deck)
            if len(self.deck) == 0:
                return
            random.shuffle(self.deck)
            self.hand.append(self.deck.pop())

    def lesson_add(self, num):
        '''
        加分
        '''
        if self.lesson_buff > 0:
            num += self.lesson_buff
        mul = 1.0
        if self.parameter_buff > 0:
            mul += 0.5
        if self.parameter_buff_multiple_per_turn > 0:
            mul += self.parameter_buff * 0.1
        num = math.ceil(num * mul)
        self.lesson += num

    def lesson_add_depend(self, percent, depend = "block", reduce = 0000):
        '''
        加分，比例依赖，由于仅logic所以不用管lesson_buff和parameter_buff
        '''
        if depend == "block":
            num = self.block * percent
        elif depend == "stamina":
            num = self.stamina * percent
        elif depend == "review":
            num = self.review * percent
        if reduce > 0:
            self.block *= (1-reduce)
        self.lesson += num

    def start_turn(self):
        '''
        开始回合
        1. 抽牌,结算timer_card_draw
        2. 结算timer_lesson
        3. 结算timer_card_upgrade
        '''
        self.card_draw = 3
        self.card_draw += self.timer_card_draw[0]
        self.timer_card_draw[0] = self.timer_card_draw[1]
        self.timer_card_draw[1] = 0

        pass

    def play_card(self,card_idx):
        '''
        打出牌
        1. 结算效果
        2. 结算是否结束回合
        '''
        pass
    
    def turn_process(self):
        '''
        回合进行
        直到不可出牌为止
        1. 打出牌
        2. 结算效果
        3. 结算是否结束回合
        '''
        pass

    def end_turn():
        '''
        结束回合
        1. 结算
        '''
        pass