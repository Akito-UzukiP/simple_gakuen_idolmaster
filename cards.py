# 卡片基础类，包含卡片的基础攻击力 attack和cost
import numpy as np
class Card:
    def __init__(self, name, effects: list[int], upgrade_effects: list[int], exile=False, upgraded=False, limit=False, first=False):
        self.name = name
        self.effects = effects
        self.exile = exile
        self.upgrade_effects = upgrade_effects
        self.upgraded = upgraded
        self.limit = limit
        self.first = first

    def __str__(self):
        string = '(' + self.name + '+: ' if self.upgraded else '(' + self.name + ': '
        string += '体力: ' + str(self.effects[0]) + ', ' 
        string += '直接体力: ' + str(self.effects[1]) + ', ' if self.effects[1] > 0 else ''
        string += '元気: ' + str(self.effects[2]) + ', ' if self.effects[2] > 0 else ''
        string += 'やる気: ' + str(self.effects[3]) + ', ' if self.effects[3] > 0 else ''
        string += '好印象: ' + str(self.effects[4]) + ', ' if self.effects[4] > 0 else ''
        string += '好印象消耗:' + str(self.effects[4]) + ', ' if self.effects[4] < 0 else ''
        string += '好調: ' + str(self.effects[5]) + ', ' if self.effects[5] > 0 else ''
        string += '絶好調: ' + str(self.effects[6]) + ', ' if self.effects[6] > 0 else ''
        string += 'パラメータ: ' + str(self.effects[7]) + ', ' if self.effects[7] > 0 else ''
        string += str(self.effects[8]) + '元気パラメータUP, ' if self.effects[8] > 0 else ''
        string += str(self.effects[9]) + '好印象パラメータUP, ' if self.effects[9] > 0 else ''
        string += '体力增加: ' + str(self.effects[10]) + ', ' if self.effects[10] > 0 else ''
        string += '体力减少: ' + str(self.effects[11]) + ', ' if self.effects[11] > 0 else ''
        string += '直接体力减少: ' + str(self.effects[12]) + ', ' if self.effects[12] > 0 else ''
        string += '额外可打出: ' + str(self.effects[13]) + ', ' if self.effects[13] > 0 else ''
        string += '额外抽牌: ' + str(self.effects[14]) + ', ' if self.effects[14] > 0 else ''
        string += '额外回合数' + str(self.effects[15]) + ', ' if self.effects[15] > 0 else ''
        string += 'レッスン中一回' if self.exile else ''
        if string.endswith(', '):
            string = string[:-2]
        string += ')'
        return string

    def __repr__(self) -> str:
        return self.__str__()

    def upgrade(self):
        new_card = Card(self.name, self.effects, self.upgrade_effects, self.exile, True, self.limit, self.first)
        for i in range(len(new_card.effects)):
            new_card.effects[i] = self.effects[i] + self.upgrade_effects[i]
            new_card.upgraded = True
        return new_card

    def observe(self):
        # 将卡片的效果转化为可微的数值
        observation = []
        observation.extend(self.effects)
        observation.append(1 if self.exile else 0)
        observation.append(1 if self.upgraded else 0)
        return np.array(observation)

def create_card(name, cost=0, direct_cost=0, robust=0, motivation=0, good_impression=0, good_condition=0, best_condition=0, score=0, score_robust_percent=0, score_good_impression_percent=0, 
                hp_damage_increase=0, hp_damage_decrease=0, hp_damage_decrease_direct=0, additional_playable=0, additional_draw=0, additional_turn = 0,
                upgrade_cost=0, upgrade_direct_cost=0, upgrade_robust=0, upgrade_motivation=0, upgrade_good_impression=0, upgrade_good_condition=0, upgrade_best_condition=0, upgrade_score=0, 
                upgrade_score_robust_percent=0, upgrade_score_good_impression_percent=0, upgrade_hp_damage_increase=0, upgrade_hp_damage_decrease=0, upgrade_hp_damage_decrease_direct=0, 
                upgrade_additional_playable=0, upgrade_additional_draw=0, upgrade_additional_turn=0, exile=False, upgraded=False, limit=False, first=False):
    return Card(name, [cost, direct_cost, robust, motivation, good_impression, good_condition, best_condition, score, score_robust_percent, score_good_impression_percent, 
                       hp_damage_increase, hp_damage_decrease, hp_damage_decrease_direct, additional_playable, additional_draw, additional_turn], 
                [upgrade_cost, upgrade_direct_cost, upgrade_robust,upgrade_motivation, upgrade_good_impression, upgrade_good_condition, upgrade_best_condition, upgrade_score, 
                 upgrade_score_robust_percent, upgrade_score_good_impression_percent, upgrade_hp_damage_increase, upgrade_hp_damage_decrease, upgrade_hp_damage_decrease_direct, 
                 upgrade_additional_playable, upgrade_additional_draw, upgrade_additional_turn], exile, upgraded, limit, first)
# effect是效果列表，cost, direct_cost, robust, motivation, good_impression, good_condition, best_condition, score, score_robust_percent, score_good_impression_percent,
# hp_damage_increase, hp_damage_decrease, hp_damage_decrease_direct, additional_playable, additional_draw, additional_turn
# 即： 体力消耗，真实体力消耗，加元气，加干劲, 加好印象，加好调，加绝好调，加分数，加分数（按元气比例），加分数（按好印象），
# 体力增加，体力减少，直接体力减少，额外可打出，额外抽牌

# 以下是卡片的具体效果
# アピールの基本
# cost4, score9
appeal_basic = create_card('アピールの基本', cost = 4, score = 9, upgrade_cost=-1, upgrade_score=5)
# ポーズの基本
#cost3, score2, robust2
pose_basic = create_card('ポーズの基本', cost = 3, score = 2, robust = 2, upgrade_score=4, upgrade_robust=2)
#表現の基本
#cost0, robust4, exile
expression_basic = create_card('表現の基本', robust = 4, exile = True, upgrade_robust=3)
# 目線の基本
# cost2, good_impression2
eye_contact_basic = create_card('目線の基本', cost = 2, good_impression = 2, robust=1, upgrade_good_impression=1, upgrade_robust=1)
# 
# 可愛い仕草
# cost5, good_impression2, score_good_imporession1.0
cute_gesture = create_card('可愛い仕草', cost = 5, good_impression = 2, score_good_impression_percent = 1.0, exile = True, upgrade_good_impression=1, upgrade_score_good_impression_percent=0.2)

# 気分転換
# direct_cost5, score_robust_percent1.0, exile
change_of_mood = create_card('気分転換', direct_cost = 5, score_robust_percent = 1.0, exile = True, upgrade_direct_cost=-2, upgrade_score_robust_percent=0.1)

# 振る舞いの基本
# cost1, good_condition2, robust1
behavior_basic = create_card('振る舞いの基本', cost = 1, good_condition = 2, robust = 1, upgrade_good_condition=1)

# 意識の基本
# cost2, motivation2, robust1
consciousness_basic = create_card('意識の基本', cost = 2, motivation = 2, robust = 1, upgrade_motivation=1, upgrade_robust=1)

# 200%スマイル
# cost6 good_impression5, score_good_impression_percent1.0
smile_200 = create_card('200%スマイル', cost = 6, good_impression = 5, score_good_impression_percent = 1.0, upgrade_good_impression=1, upgrade_score_good_impression_percent=0.7)

# ふれあい
# cost5, good_impression4, robust3, exile
touch = create_card('ふれあい', cost = 5, good_impression = 4, robust = 3, exile = True, upgrade_good_impression=1, upgrade_robust=3)

# ラブリーウインク
# cost5, good_impression4, score_good_impression_percent0.6
lovely_wink = create_card('ラブリーウインク', cost = 5, good_impression = 4, score_good_impression_percent = 0.6, upgrade_score_good_impression_percent=0.2, upgrade_good_impression=1)

# リズミカル
# cost0, robust6
rhythmic = create_card('リズミカル', robust = 6, upgrade_robust=2, exile=True)

#　幸せな時間
# cost5, good_impression6
happy_time = create_card('幸せな時間', cost = 5, good_impression = 6, upgrade_good_impression=2)

# リスタート
# cost4, good_impression3
restart = create_card('リスタート', cost = 4, good_impression = 3, robust=2, upgrade_good_impression=1, upgrade_cost=-1)

# キラメキ
# cost3, score_good_impression_percent2.0, 体力消耗增加2turn (WIP)
kirameki = create_card('キラメキ', cost = 3, score_good_impression_percent = 2.0, hp_damage_increase = 2, upgrade_score_good_impression_percent=0.5)
# 本番前夜
# cost5, good_impression4, motivation3, 升级++1，开局必得
honbanzenya = create_card('本番前夜', cost = 5, good_impression = 4, motivation = 3, upgrade_good_impression = 1, upgrade_motivation=1,first = True)

# 私がスター
# cost0, good_impression-1, additional_turn1, additional_playable1, 升级 additional_draw1
watashi_ga_star = create_card('私がスター', good_impression = -1, additional_turn = 1, additional_playable = 1, upgrade_additional_draw = 1, limit=True)

# 手拍子
# cost5, score_good_impression_percent1.5, upgrade_score_good_impression_percent0.5, exile
clap = create_card('手拍子', cost = 5, score_good_impression_percent = 1.5, upgrade_score_good_impression_percent=0.5, exile=True)

# やる気は満点
# cost1, robust1, good_impression4, exile, upgrade_good_impression1, upgrade_robust1
good_condition = create_card('やる気は満点', cost = 1, robust = 1, good_impression = 4, exile = True, upgrade_good_impression=1, upgrade_robust=1)

# アイドル宣言
# direct_cost1, additional_playable1, additional_draw2, upgrade_direct_cost-1, upgrade_hp_damage_decrease1
idol_declaration = create_card('アイドル宣言', direct_cost = 1, additional_playable = 1, additional_draw = 2, upgrade_direct_cost=-1, upgrade_hp_damage_decrease=1, limit=True,exile=True)

# テレビ出演
# cost1, robust3, hp_damage_decrease4, upgrade_robust2, upgrade_hp_damage_decrease1, limit, exile
tv_show = create_card('テレビ出演', cost = 1, robust = 3, hp_damage_decrease = 4, upgrade_robust=2, upgrade_hp_damage_decrease=1, limit=True,exile=True)

# 星屑センセーション
# motivation-3,good_impression5, additional_playable1, upgrade_good_impression2, upgrade_additional_draw1, limit, exile
# stardust_sensation = create_card('星屑センセーション', motivation = -3, good_impression = 5, additional_playable = 1, upgrade_good_impression=2, upgrade_additional_draw=1, limit=True,exile=True) 

# 休憩
# cost-2
rest = create_card('休憩', cost = -2, limit = True)


# 以下是特殊卡片
# ktn的ssr
# よそ見はだめ
# cost6, good_impression7, exile
ktn_ssr = create_card('よそ見はだめ♪', cost = 6, good_impression = 7, exile = True, upgrade_good_impression=2, upgrade_robust=2, limit = True)

def create_ktn_deck():
    return [appeal_basic]*2 + [pose_basic]*1 + [expression_basic]*2 + [eye_contact_basic]*2 + [cute_gesture]*1 + [ktn_ssr]*1


# 特殊效果：
# 多次触发加分
# 

#all_cards = [clap, appeal_basic, pose_basic, expression_basic, eye_contact_basic, cute_gesture, change_of_mood, behavior_basic, consciousness_basic, smile_200, touch, lovely_wink, rhythmic, happy_time, restart, kirameki, honbanzenya, watashi_ga_star, ktn_ssr]
all_cards = [clap, appeal_basic, pose_basic, expression_basic, eye_contact_basic, cute_gesture, change_of_mood, behavior_basic, consciousness_basic, smile_200, touch, lovely_wink, rhythmic, happy_time, restart, kirameki, honbanzenya, watashi_ga_star, idol_declaration, tv_show, good_condition, rest]
upgraded_cards = [card.upgrade() for card in all_cards]

# 随机ktn卡组，包含基础ktn卡组加上15张随机卡，其中有5张升级卡，limit卡仅能有一张
import random
def create_random_ktn_deck():
    deck = create_ktn_deck()
    for i in range(10):
        card = random.choice(all_cards)
        if card.limit and all_cards[i] not in deck and upgraded_cards[i] not in deck:
            while card in deck:
                card = random.choice(all_cards)
        deck.append(card)
    for i in range(5):
        card = random.choice(upgraded_cards)
        if card.limit and all_cards[i] not in deck and upgraded_cards[i] not in deck:
            while card in deck:
                card = random.choice(upgraded_cards)
        deck.append(card)
    return deck
#print(create_random_ktn_deck())
#print(create_card('ラブリーウインク', cost = 5, good_impression = 4, score_good_impression_percent = 0.6, upgrade_score_good_impression_percent=0.2).upgrade())