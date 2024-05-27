# 卡片基础类，包含卡片的基础攻击力 attack和cost

class Card:
    def __init__(self, name, effects: list[int], upgrade_effects:list[int], exile = False, upgraded = False):
        self.name = name
        self.effects = effects
        self.exile = exile
        self.upgrade_effects = upgrade_effects
        self.upgraded = upgraded
    def __str__(self):
        string = '(' + self.name + '+: ' if self.upgraded else '('+self.name + ': '
        string += '体力: ' + str(self.effects[0]) + ', ' 
        string += '直接体力: ' + str(self.effects[1]) + ', ' if self.effects[1] > 0 else ''
        string += '元気: ' + str(self.effects[2]) + ', ' if self.effects[2] > 0 else ''
        string += '好印象: ' + str(self.effects[3]) + ', ' if self.effects[3] > 0 else ''
        string += '好調: ' + str(self.effects[4]) + ', ' if self.effects[4] > 0 else ''
        string += '絶好調: ' + str(self.effects[5]) + ', ' if self.effects[5] > 0 else ''
        string += 'パラメータ: ' + str(self.effects[6]) + ', ' if self.effects[6] > 0 else ''
        string +=   str(self.effects[7]) + '元気パラメータUP, ' if self.effects[7] > 0 else ''
        string += str(self.effects[8]) + '好印象パラメータUP, ' if self.effects[8] > 0 else ''
        string += 'レッスン中一回' if self.exile else ''
        if string.endswith(', '):
            string = string[:-2]
        string += ')'
        return string
    def __repr__(self) -> str:
        return self.__str__()
    def upgrade(self):
        self.effects = self.upgrade_effects
        self.upgraded = True

def create_card(name, cost = 0, direct_cost = 0, robust = 0, good_impression = 0, good_condition = 0, best_condition = 0, score = 0, score_robust_percent = 0, score_good_impression_percent = 0, 
                upgrade_cost = 0, upgrade_direct_cost = 0, upgrade_robust = 0, upgrade_good_impression = 0, upgrade_good_condition = 0, upgrade_best_condition = 0, upgrade_score = 0, upgrade_score_robust_percent = 0, upgrade_score_good_impression_percent = 0,
                exile = False, upgraded = False):
    return Card(name, [cost, direct_cost, robust, good_impression, good_condition, best_condition, score, score_robust_percent, score_good_impression_percent], 
                [upgrade_cost, upgrade_direct_cost, upgrade_robust, upgrade_good_impression, upgrade_good_condition, upgrade_best_condition, upgrade_score, 
                 upgrade_score_robust_percent, upgrade_score_good_impression_percent], exile, upgraded)
# effect是效果列表，cost, direct_cost, robust, good_impression, good_condition, best_condition, score, score_robust_percent, score_good_impression_percent
# 即： 体力消耗，真实体力消耗，加元气，加好印象，加好调，加绝好调，加分数，加分数（按元气比例），加分数（按好印象）

# 以下是卡片的具体效果
# アピールの基本
# cost4, score9
appeal_basic = create_card('アピールの基本', cost = 4, score = 9)
# ポーズの基本
#cost3, score2, robust2
pose_basic = create_card('ポーズの基本', cost = 3, score = 2, robust = 2)
#表現の基本
#cost0, robust4, exile
expression_basic = create_card('表現の基本', robust = 4, exile = True)
# 目線の基本
# cost2, good_impression2
eye_contact_basic = create_card('目線の基本', cost = 2, good_impression = 2)
# 
# 可愛い仕草
# cost5, good_impression2, score_good_imporession100
cute_gesture = create_card('可愛い仕草', cost = 5, good_impression = 2, score_good_impression_percent = 1.0, exile = True)


# 以下是特殊卡片
# ktn的ssr
# よそ見はだめ
# cost6, good_impression7, exile
ktn_ssr = create_card('よそ見はだめ♪', cost = 6, good_impression = 7, exile = True)

def create_ktn_deck():
    return [appeal_basic]*2 + [pose_basic]*1 + [expression_basic]*2 + [eye_contact_basic]*2 + [cute_gesture]*1 + [ktn_ssr]*1