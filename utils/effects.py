import re
import math
try:
    from . import triggers
except:
    import utils.triggers as triggers
from typing import Any
# e_effect-exam_block-(\d+) ->元气增加
exam_block = re.compile(r"e_effect-exam_block-(\d+)$")
def match_exam_block(string):
    default_values = [0]
    # 如果匹配成功，返回匹配的值，否则返回默认值
    if exam_block.match(string):
        return [int(i) for i in exam_block.match(string).groups(default_values)]
    return default_values
# e_effect-exam_card_play_aggressive-(\d+) -> やる気增加
exam_card_play_aggressive = re.compile(r"e_effect-exam_card_play_aggressive-(\d+)$")
def match_exam_card_play_aggressive(string):
    default_values = [0]
    if exam_card_play_aggressive.match(string):
        return [int(i) for i in exam_card_play_aggressive.match(string).groups(default_values)]
    return default_values
# e_effect-exam_lesson-[0-9]-(\d+) -> 加分-加分次数
exam_lesson = re.compile(r"e_effect-exam_lesson-(\d+)-(\d+)$")
def match_exam_lesson(string):
    default_values = [0, 0]
    if exam_lesson.match(string):
        return [int(i) for i in exam_lesson.match(string).groups(default_values)]
    return default_values
# e_effect-exam_review-(\d+) -> 好印象增加
exam_review = re.compile(r"e_effect-exam_review-(\d+)$")
def match_exam_review(string):
    default_values = [0]
    if exam_review.match(string):
        return [int(i) for i in exam_review.match(string).groups(default_values)]
    return default_values
# e_effect-exam_lesson_depend_block-(\d+)-(\d+) -> 依赖元气千分比加分，元气千分比减少，加分次数
# e_effect-exam_lesson_depend_block-(\d+)-(\d+)-(\d+)$ -> 依赖元气千分比加分，加分次数
exam_lesson_depend_block_decrease = re.compile(r"e_effect-exam_lesson_depend_block-(\d+)-(\d+)-(\d+)$")
exam_lesson_depend_block = re.compile(r"e_effect-exam_lesson_depend_block-(\d+)-(\d+)$")
def match_exam_lesson_depend_block(string):
    default_values = [0, 0, 0]
    if exam_lesson_depend_block_decrease.match(string):
        return [int(i) for i in exam_lesson_depend_block_decrease.match(string).groups(default_values)]
    else:
        if exam_lesson_depend_block.match(string):
            tmp = [int(i) for i in exam_lesson_depend_block.match(string).groups(default_values)]
            return [tmp[0], 0, tmp[1]]
    return default_values

# e_effect-exam_lesson_depend_review-(\d+)-(\d+) -> 依赖好印象加分，加分次数
exam_lesson_depend_review = re.compile(r"e_effect-exam_lesson_depend_review-(\d+)-(\d+)$")
def match_exam_lesson_depend_review(string):
    default_values = [0, 0]
    if exam_lesson_depend_review.match(string):
        return [int(i) for i in exam_lesson_depend_review.match(string).groups(default_values)]
    return default_values
# e_effect-exam_stamina_consumption_add-(\d+) -> 体力消耗增加，增加回合数
exam_stamina_consumption_add = re.compile(r"e_effect-exam_stamina_consumption_add-(\d+)$")
def match_exam_stamina_consumption_add(string):
    default_values = [0]
    if exam_stamina_consumption_add.match(string):
        return [int(i) for i in exam_stamina_consumption_add.match(string).groups(default_values)]
    return default_values
# e_effect-exam_stamina_consumption_down-(\d+) -> 体力消耗减少，减少回合数
exam_stamina_consumption_down = re.compile(r"e_effect-exam_stamina_consumption_down-(\d+)$")
def match_exam_stamina_consumption_down(string):
    default_values = [0]
    if exam_stamina_consumption_down.match(string):
        return [int(i) for i in exam_stamina_consumption_down.match(string).groups(default_values)]
    return default_values
# e_effect-exam_stamina_consumption_down_fix-(\d+)-inf -> 固定减少体力消耗
exam_stamina_consumption_down_fix = re.compile(r"e_effect-exam_stamina_consumption_down_fix-(\d+)-inf")
def match_exam_stamina_consumption_down_fix(string):
    default_values = [0]
    if exam_stamina_consumption_down_fix.match(string):
        return [int(i) for i in exam_stamina_consumption_down_fix.match(string).groups(default_values)]
    return default_values
# e_effect-exam_stamina_recover_fix-(\d+) -> 固定回复体力，单次
exam_stamina_recover_fix_once = re.compile(r"e_effect-exam_stamina_recover_fix-(\d+)$")
def match_exam_stamina_recover_fix_once(string):
    default_values = [0]
    if exam_stamina_recover_fix_once.match(string):
        return [int(i) for i in exam_stamina_recover_fix_once.match(string).groups(default_values)]
    return default_values
#  e_effect-exam_parameter_buff-(\d+) -> 好调增加，回合数
exam_parameter_buff = re.compile(r"e_effect-exam_parameter_buff-(\d+)$")
def match_exam_parameter_buff(string):
    default_values = [0]
    if exam_parameter_buff.match(string):
        return [int(i) for i in exam_parameter_buff.match(string).groups(default_values)]
    return default_values
# e_effect-exam_lesson_buff-(\d+) -> 集中增加，回合数
exam_lesson_buff = re.compile(r"e_effect-exam_lesson_buff-(\d+)$")
def match_exam_lesson_buff(string):
    default_values = [0]
    if exam_lesson_buff.match(string):
        return [int(i) for i in exam_lesson_buff.match(string).groups(default_values)]
    return default_values
# e_effect-exam_multiple_lesson_buff_lesson-(\d+)-(\d+)-(\d+) -> 集中倍率增加千分数，回合数，次数，对于卡本身来说
exam_multiple_lesson_buff_lesson = re.compile(r"e_effect-exam_multiple_lesson_buff_lesson-(\d+)-(\d+)-(\d+)$")
def match_exam_multiple_lesson_buff_lesson(string):
    default_values = [0, 0, 0]
    if exam_multiple_lesson_buff_lesson.match(string):
        return [int(i) for i in exam_multiple_lesson_buff_lesson.match(string).groups(default_values)]
    return default_values
# e_effect-exam_lesson_depend_exam_card_play_aggressive-(\d+)-(\d+) -> 依赖やる気加分，加分次数
exam_lesson_depend_exam_card_play_aggressive = re.compile(r"e_effect-exam_lesson_depend_exam_card_play_aggressive-(\d+)-(\d+)$")
def match_exam_lesson_depend_exam_card_play_aggressive(string):
    default_values = [0, 0]
    if exam_lesson_depend_exam_card_play_aggressive.match(string):
        return [int(i) for i in exam_lesson_depend_exam_card_play_aggressive.match(string).groups(default_values)]
    return default_values
# e_effect-exam_parameter_buff_multiple_per_turn-(\d+) -> 绝好调回合数
exam_parameter_buff_multiple_per_turn = re.compile(r"e_effect-exam_parameter_buff_multiple_per_turn-(\d+)$")
def match_exam_parameter_buff_multiple_per_turn(string):
    default_values = [0]
    if exam_parameter_buff_multiple_per_turn.match(string):
        return [int(i) for i in exam_parameter_buff_multiple_per_turn.match(string).groups(default_values)]
    return default_values
# e_effect-exam_lesson_depend_exam_review-(\d+)-(\d+) -> 依赖好印象加分，加分次数
exam_lesson_depend_exam_review = re.compile(r"e_effect-exam_lesson_depend_exam_review-(\d+)-(\d+)$")
def match_exam_lesson_depend_exam_review(string):
    default_values = [0, 0]
    if exam_lesson_depend_exam_review.match(string):
        return [int(i) for i in exam_lesson_depend_exam_review.match(string).groups(default_values)]
    return default_values
# e_effect-exam_effect_timer-(\d+)-(\d+)-e_effect-exam_card_draw-(\d+) -> 效果回合数，发动次数，抽卡次数
exam_effect_timer_draw = re.compile(r"e_effect-exam_effect_timer-(\d+)-(\d+)-e_effect-exam_card_draw-(\d+)$")
def match_exam_effect_timer_draw(string):
    default_values = [0, 0, 0]
    if exam_effect_timer_draw.match(string):
        return [int(i) for i in exam_effect_timer_draw.match(string).groups(default_values)]
    return default_values
# e_effect-exam_effect_timer-(\d+)-(\d+)-e_effect-exam_lesson-(\d+)-(\d+) -> 效果回合数，发动次数，加分，加分次数
exam_effect_timer_lesson = re.compile(r"e_effect-exam_effect_timer-(\d+)-(\d+)-e_effect-exam_lesson-(\d+)-(\d+)$")
def match_exam_effect_timer_lesson(string):
    default_values = [0, 0, 0, 0]
    if exam_effect_timer_lesson.match(string):
        return [int(i) for i in exam_effect_timer_lesson.match(string).groups(default_values)]
    return default_values
# e_effect-exam_effect_timer-(\d+)-(\d+)-e_effect-exam_card_upgrade-p_card_search-hand-all-0_0 -> 效果回合数，发动次数，升级所有手牌
exam_effect_timer_card_upgrade = re.compile(r"e_effect-exam_effect_timer-(\d+)-(\d+)-e_effect-exam_card_upgrade-p_card_search-hand-all-0_0")
def match_exam_effect_timer_card_upgrade(string):
    default_values = [0, 0]
    if exam_effect_timer_card_upgrade.match(string):
        return [int(i) for i in exam_effect_timer_card_upgrade.match(string).groups(default_values)]
    return default_values
# e_effect-exam_extra_turn -> 额外回合
exam_extra_turn = re.compile(r"e_effect-exam_extra_turn")
def match_exam_extra_turn(string):
    default_values = [0]
    if exam_extra_turn.match(string):
        return [1]
    return default_values
# e_effect-exam_playable_value_add-(\d+) -> 出牌数增加
exam_playable_value_add = re.compile(r"e_effect-exam_playable_value_add-(\d+)$")
def match_exam_playable_value_add(string):
    default_values = [0]
    if exam_playable_value_add.match(string):
        return [int(i) for i in exam_playable_value_add.match(string).groups(default_values)]
    return default_values
# e_effect-exam_hand_grave_count_card_draw 手牌去墓地，同样数量牌上手
exam_hand_grave_count_card_draw = re.compile(r"e_effect-exam_hand_grave_count_card_draw")
def match_exam_hand_grave_count_card_draw(string):
    default_values = [0]
    if exam_hand_grave_count_card_draw.match(string):
        return [1]
    return default_values
# e_effect-exam_card_draw-(\d+) -> 抽卡次数
exam_card_draw = re.compile(r"e_effect-exam_card_draw-(\d+)$")
def match_exam_card_draw(string):
    default_values = [0]
    if exam_card_draw.match(string):
        return [int(i) for i in exam_card_draw.match(string).groups(default_values)]
    return default_values
# e_effect-exam_card_create_search-0001-p_card_search-random-random_pool-p_random_pool-all-upgrade_1-1-hand-random-1_1 -> 上手升级后的卡
exam_card_create_search = re.compile(r"e_effect-exam_card_create_search-0001-p_card_search-random-random_pool-p_random_pool-all-upgrade_1-1-hand-random-1_1")
def match_exam_card_create_search(string):
    default_values = [0]
    if exam_card_create_search.match(string):
        return [1]
    return default_values
# e_effect-exam_anti_debuff-(\d+) -> 负面效果无效回合数
exam_anti_debuff = re.compile(r"e_effect-exam_anti_debuff-(\d+)$")
def match_exam_anti_debuff(string):
    default_values = [0]
    if exam_anti_debuff.match(string):
        return [int(i) for i in exam_anti_debuff.match(string).groups(default_values)]
    return default_values
# e_effect-exam_block_restriction-(\d+) -> 元气增加无效回合数
exam_block_restriction = re.compile(r"e_effect-exam_block_restriction-(\d+)$")
def match_exam_block_restriction(string):
    default_values = [0]
    if exam_block_restriction.match(string):
        return [int(i) for i in exam_block_restriction.match(string).groups(default_values)]
    return default_values
# e_effect-exam_card_search_effect_play_count_buff-0001-01-inf-p_card_search-playing-all-0_0 -> 下一张卡效果发动2次
exam_card_search_effect_play_count_buff = re.compile(r"e_effect-exam_card_search_effect_play_count_buff-0001-01-inf-p_card_search-playing-all-0_0")
def match_exam_card_search_effect_play_count_buff(string):
    default_values = [0]
    if exam_card_search_effect_play_count_buff.match(string):
        return [1]
    return default_values
# e_effect-exam_card_upgrade-p_card_search-hand-all-0_0 -> 升级所有手牌
exam_card_upgrade = re.compile(r"e_effect-exam_card_upgrade-p_card_search-hand-all-0_0")
def match_exam_card_upgrade(string):
    default_values = [0]
    if exam_card_upgrade.match(string):
        return [1]
    return default_values
# 特殊持续效果
# e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_034-enc01 以降、 アクティブスキルカード 使用時、 固定元気  +2  0
# e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_038-enc01 以降、 アクティブスキルカード 使用時、 集中  +1 1

# e_effect-exam_status_enchant-inf-enchant-p_card-01-men-3_035-enc01 以降、ターン終了時 集中 が3以上の場合、 集中  +2 2
# e_effect-exam_status_enchant-inf-enchant-p_card-02-ido-1_016-enc01 以降、ターン終了時、 好印象  +1 3
# e_effect-exam_status_enchant-inf-enchant-p_card-02-ido-3_019-enc01 以降、ターン終了時、 好印象  +1 4
# e_effect-exam_status_enchant-inf-enchant-p_card-02-men-2_004-enc01 以降、 メンタルスキルカード 使用時、 やる気  +1 5
# e_effect-exam_status_enchant-inf-enchant-p_card-02-men-2_058-enc01 以降、 メンタルスキルカード 使用時、 好印象  +1 6
# e_effect-exam_status_enchant-inf-enchant-p_card-02-men-3_042-enc01 以降、ターン終了時 好印象 が3以上の場合、 好印象  +3 7
e_effects = [
    "e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_034-enc01",
    "e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_038-enc01",
    "e_effect-exam_status_enchant-inf-enchant-p_card-01-men-3_035-enc01",
    "e_effect-exam_status_enchant-inf-enchant-p_card-02-ido-1_016-enc01",
    "e_effect-exam_status_enchant-inf-enchant-p_card-02-ido-3_019-enc01",
    "e_effect-exam_status_enchant-inf-enchant-p_card-02-men-2_004-enc01",
    "e_effect-exam_status_enchant-inf-enchant-p_card-02-men-2_058-enc01",
    "e_effect-exam_status_enchant-inf-enchant-p_card-02-men-3_042-enc01",
]

def match_e_effects(string):
    for i,pattern in enumerate(e_effects):
        if pattern in string:
            #print(pattern)
            return [i+1]
    return [0]

all_res = [exam_block, exam_card_play_aggressive, exam_lesson, exam_review, 
           exam_lesson_depend_block, 
           exam_stamina_consumption_add, exam_stamina_consumption_down, exam_stamina_consumption_down_fix, 
            exam_stamina_recover_fix_once, exam_parameter_buff, exam_lesson_buff, 
           exam_multiple_lesson_buff_lesson, exam_lesson_depend_exam_card_play_aggressive,exam_parameter_buff_multiple_per_turn, 
           exam_lesson_depend_exam_review, exam_effect_timer_draw, exam_effect_timer_lesson, exam_effect_timer_card_upgrade,
           exam_extra_turn, exam_playable_value_add, exam_hand_grave_count_card_draw, exam_card_draw, exam_card_create_search,
           exam_anti_debuff, exam_block_restriction, exam_card_search_effect_play_count_buff, exam_card_upgrade] + [re.compile(i) for i in e_effects]


def match_all_effects(string):
    all_results = []

    # 调用所有的匹配函数并将结果连接起来
    all_results.append(match_exam_block(string))# 元气增加
    all_results.append(match_exam_card_play_aggressive(string))# やる気增加
    all_results.append(match_exam_lesson(string))# 加分
    all_results.append(match_exam_review(string))# 好印象增加
    all_results.append(match_exam_lesson_depend_block(string))# 依赖元气加分
    all_results.append(match_exam_stamina_consumption_add(string))# 体力消耗增加
    all_results.append(match_exam_stamina_consumption_down(string))# 体力消耗减少
    all_results.append(match_exam_stamina_consumption_down_fix(string))# 固定体力消耗减少
    all_results.append(match_exam_stamina_recover_fix_once(string))# 固定回复体力，单次
    all_results.append(match_exam_parameter_buff(string))# 好调增加
    all_results.append(match_exam_lesson_buff(string))# 集中增加
    all_results.append(match_exam_multiple_lesson_buff_lesson(string))# 集中倍率增加
    all_results.append(match_exam_lesson_depend_exam_card_play_aggressive(string))# 依赖やる気加分
    all_results.append(match_exam_parameter_buff_multiple_per_turn(string))# 绝好调回合数
    all_results.append(match_exam_lesson_depend_exam_review(string))# 依赖好印象加分
    all_results.append(match_exam_effect_timer_draw(string))# 后续回合，抽卡
    all_results.append(match_exam_effect_timer_lesson(string))# 后续回合，加分
    all_results.append(match_exam_effect_timer_card_upgrade(string))# 后续回合，升级所有手牌
    all_results.append(match_exam_extra_turn(string))# 增加回合
    all_results.append(match_exam_playable_value_add(string))# 出牌数增加
    all_results.append(match_exam_hand_grave_count_card_draw(string))# 手牌去墓地，同样数量牌上手
    all_results.append(match_exam_card_draw(string))# 立刻抽卡
    all_results.append(match_exam_card_create_search(string))# 上手一张升级后的卡
    all_results.append(match_exam_anti_debuff(string))# 负面效果无效回合数
    all_results.append(match_exam_block_restriction(string))# 元气增加无效回合数
    all_results.append(match_exam_card_search_effect_play_count_buff(string))# 下一张卡效果发动2次
    all_results.append(match_exam_card_upgrade(string))# 升级所有手牌
    all_results.append(match_e_effects(string))# 特殊效果

    return all_results





#print(len(match_all_effects("e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_034-enc01")))

# 回合开始相关结算
# 1. 各触发时点在开始的timer结算(额外抽卡、额外加分、额外升级手牌、道具效果)
# 2. 抽卡

# 牌打出相关结算
# 1. 体力结算
# 2. 其它消费结算
# 3. 元气增加
# 8. やる気增加
# 9. 普通加分
# 11. 集中倍率加分、多次加分
# 7. 好印象增加
# 5. 好调增加
# 6. 绝好调增加
# 4. 集中增加
# 10. 好印象、好调、やる気、元气倍率加分
## ↑基础部分
## ↓特殊部分
# 12. 体力消耗增加
# 13. 体力消耗减少
# 14. 直接体力消耗减少
# 15. 额外抽牌
# 16. 额外回合数
# 17. 额外出牌数
# 18. 后续回合，抽卡
# 19. 后续回合，加分
# 20. 后续回合，升级所有手牌
# 21. 下一张卡效果发动2次
# 22. 上手一张SSR+
# 23. 负面效果无效回合数
# 24. 元气增加无效回合数
# 25. 增加永续回调效果，当使用x卡牌时，效果发动
# 26. 换牌
# 27. 回复体力（？没见过）

# 回合结束相关结算
# 1. 好调、绝好调、好印象、体力消耗增加、体力消耗减少 timer减少 （若该timer在当回合由0变为非0，则当回合不消耗）




class Effect:
    def __init__(self):
        self.exam_block = [0]
        self.exam_card_play_aggressive = [0]
        self.exam_lesson = [0, 0]
        self.exam_review = [0]
        self.exam_lesson_depend_block = [0,0,0]
        self.exam_stamina_consumption_add = [0]
        self.exam_stamina_consumption_down = [0]
        self.exam_stamina_consumption_down_fix = [0]
        self.exam_stamina_recover_fix_once = [0]
        self.exam_parameter_buff = [0]
        self.exam_lesson_buff = [0]
        self.exam_multiple_lesson_buff_lesson = [0, 0, 0]
        self.exam_lesson_depend_exam_card_play_aggressive = [0, 0]
        self.exam_parameter_buff_multiple_per_turn = [0]
        self.exam_lesson_depend_exam_review = [0, 0]
        self.exam_effect_timer_draw = [0, 0, 0]
        self.exam_effect_timer_lesson = [0, 0, 0, 0]
        self.exam_effect_timer_card_upgrade = [0, 0]
        self.exam_extra_turn = [0]
        self.exam_playable_value_add = [0]
        self.exam_hand_grave_count_card_draw = [0]
        self.exam_card_draw = [0]
        self.exam_card_create_search = [0]
        self.exam_anti_debuff = [0]
        self.exam_block_restriction = [0]
        self.exam_search_effect_play_count_buff = [0]
        self.exam_card_upgrade = [0]

        self.exam_status_enchant = [0] * 8

        self.card_play_trigger = [0] * len(triggers.card_play_trigger)


    def set_effect(self, effects: list[int], triggers: list[int]):
        self.exam_block = effects[0]
        self.exam_card_play_aggressive = effects[1]
        self.exam_lesson = effects[2]
        self.exam_review = effects[3]
        self.exam_lesson_depend_block = effects[4]
        # self.exam_lesson_depend_aggressive = effects[5]  # 去掉这一行
        self.exam_stamina_consumption_add = effects[5]
        self.exam_stamina_consumption_down = effects[6]
        self.exam_stamina_consumption_down_fix = effects[7]
        self.exam_stamina_recover_fix_once = effects[8]
        self.exam_parameter_buff = effects[9]
        self.exam_lesson_buff = effects[10]
        self.exam_multiple_lesson_buff_lesson = effects[11]
        self.exam_lesson_depend_exam_card_play_aggressive = effects[12]
        self.exam_parameter_buff_multiple_per_turn = effects[13]
        self.exam_lesson_depend_exam_review = effects[14]
        self.exam_effect_timer_draw = effects[15]
        self.exam_effect_timer_lesson = effects[16]
        self.exam_effect_timer_card_upgrade = effects[17]
        self.exam_extra_turn = effects[18]
        self.exam_playable_value_add = effects[19]
        self.exam_hand_grave_count_card_draw = effects[20]
        self.exam_card_draw = effects[21]
        self.exam_card_create_search = effects[22]
        self.exam_anti_debuff = effects[23]
        self.exam_block_restriction = effects[24]
        self.exam_search_effect_play_count_buff = effects[25]
        self.exam_card_upgrade = effects[26]
        self.exam_status_enchant = effects[27]


        self.card_play_trigger = triggers

    def observe(self):
        # 返回所有属性的cat
        _obs = []
        _obs.extend(self.exam_block)
        _obs.extend(self.exam_card_play_aggressive)
        _obs.extend(self.exam_lesson)
        _obs.extend(self.exam_review)
        _obs.extend(self.exam_lesson_depend_block)
        _obs.extend(self.exam_stamina_consumption_add)
        _obs.extend(self.exam_stamina_consumption_down)
        _obs.extend(self.exam_stamina_consumption_down_fix)
        _obs.extend(self.exam_stamina_recover_fix_once)
        _obs.extend(self.exam_parameter_buff)
        _obs.extend(self.exam_lesson_buff)
        _obs.extend(self.exam_multiple_lesson_buff_lesson)
        _obs.extend(self.exam_lesson_depend_exam_card_play_aggressive)
        _obs.extend(self.exam_parameter_buff_multiple_per_turn)
        _obs.extend(self.exam_lesson_depend_exam_review)
        _obs.extend(self.exam_effect_timer_draw)
        _obs.extend(self.exam_effect_timer_lesson)
        _obs.extend(self.exam_effect_timer_card_upgrade)
        _obs.extend(self.exam_extra_turn)
        _obs.extend(self.exam_playable_value_add)
        _obs.extend(self.exam_hand_grave_count_card_draw)
        _obs.extend(self.exam_card_draw)
        _obs.extend(self.exam_card_create_search)
        _obs.extend(self.exam_anti_debuff)
        _obs.extend(self.exam_block_restriction)
        _obs.extend(self.exam_search_effect_play_count_buff)
        _obs.extend(self.exam_card_upgrade)
        _obs.extend(self.exam_status_enchant)
        _obs.extend(self.card_play_trigger)
        return _obs
    
    def __str__(self):
        str_ = ""
        # trigger
        if sum(self.card_play_trigger) > 0:
            # "e_trigger-exam_card_play-card_play_aggressive_up-3",
            # "e_trigger-exam_card_play-card_play_aggressive_up-6",
            # "e_trigger-exam_card_play-lesson_buff_up-3",
            # "e_trigger-exam_card_play-lesson_buff_up-6",
            # "e_trigger-exam_card_play-parameter_buff",
            # "e_trigger-exam_card_play-review_up-1",
            # "e_trigger-exam_card_play-review_up-3",
            # "e_trigger-exam_card_play-stamina_up_multiple-500",
            str_ += " やる気 が3以上の場合、" if self.card_play_trigger[0] > 0 else ""
            str_ += " やる気 が6以上の場合、" if self.card_play_trigger[1] > 0 else ""
            str_ += " 集中 が3以上の場合、" if self.card_play_trigger[2] > 0 else ""
            str_ += " 集中 が6以上の場合、" if self.card_play_trigger[3] > 0 else ""
            str_ += " 好調　の　場合、" if self.card_play_trigger[4] > 0 else ""
            str_ += " 好印象 が1以上の場合、" if self.card_play_trigger[5] > 0 else ""
            str_ += " 好印象 が3以上の場合、" if self.card_play_trigger[6] > 0 else ""
            str_ += " 体力が50%以上の場合、" if self.card_play_trigger[7] > 0 else ""

        str_ += "元気 + " + str(self.exam_block[0]) + "\n" if sum(self.exam_block) > 0 else ""
        str_ += "やる気 + " + str(self.exam_card_play_aggressive[0]) + "\n" if sum(self.exam_card_play_aggressive) > 0 else ""
        str_ += "パラメータ + " + str(self.exam_lesson[0]) + ("、" + str(self.exam_lesson[1]) + " 回\n" if self.exam_lesson[1] > 1 else "\n") if sum(self.exam_lesson) > 0 else ""
        str_ += "好印象 + " + str(self.exam_review[0]) + "\n" if sum(self.exam_review) > 0 else ""
        # 
        str_ += "元気の " + str(self.exam_lesson_depend_block[0]/10) + " %分パラメータ上昇" + (str(self.exam_lesson_depend_block[2]) + " 回\n" if self.exam_lesson_depend_block[2] > 1 else "\n") if sum(self.exam_lesson_depend_block) > 0 else ""
        str_ += "その後、元気を " + str(self.exam_lesson_depend_block[1]/10) + " %減少\n" if self.exam_lesson_depend_block[1] > 0 else ""
        str_ += "体力消耗UP + " + str(self.exam_stamina_consumption_add[0]) + "ターン\n" if sum(self.exam_stamina_consumption_add) > 0 else ""
        str_ += "体力消耗DOWN + " + str(self.exam_stamina_consumption_down[0]) + "ターン\n" if sum(self.exam_stamina_consumption_down) > 0 else ""
        str_ += "直接体力消耗DOWN + " + str(self.exam_stamina_consumption_down_fix[0]) + "\n" if sum(self.exam_stamina_consumption_down_fix) > 0 else ""
        str_ += "体力回復 + " + str(self.exam_stamina_recover_fix_once[0]) + "\n" if sum(self.exam_stamina_recover_fix_once) > 0 else ""
        str_ += "好調 + " + str(self.exam_parameter_buff[0]) + "ターン\n" if sum(self.exam_parameter_buff) > 0 else ""
        str_ += "集中 + " + str(self.exam_lesson_buff[0]) + "\n" if sum(self.exam_lesson_buff) > 0 else ""
        str_ += "パラメータ + " + str(self.exam_multiple_lesson_buff_lesson[0]) + "、" + (str(self.exam_multiple_lesson_buff_lesson[2]) + "回" if self.exam_multiple_lesson_buff_lesson[2] > 1 else "") + " ( 集中 効果を" + str(self.exam_multiple_lesson_buff_lesson[1]/1000 + 1) +"倍 適用)\n" if sum(self.exam_multiple_lesson_buff_lesson) > 0 else ""
        #  やる気 の 200% 分 パラメータ 上昇
        str_ += "やる気の " + str(self.exam_lesson_depend_exam_card_play_aggressive[0]/10) + "%分パラメータ上昇、" + (str(self.exam_lesson_depend_exam_card_play_aggressive[1]) + "回\n" if self.exam_lesson_depend_exam_card_play_aggressive[1] > 1 else "\n") if sum(self.exam_lesson_depend_exam_card_play_aggressive) > 0 else ""
        str_ += "絶好調 + " + str(self.exam_parameter_buff_multiple_per_turn[0]) + "\n" if sum(self.exam_parameter_buff_multiple_per_turn) > 0 else ""
        str_ += "好印象の " + str(self.exam_lesson_depend_exam_review[0]/10) + "%を増加、" + (str(self.exam_lesson_depend_exam_review[1]) + "回\n" if self.exam_lesson_depend_exam_review[1] > 1 else "\n") if sum(self.exam_lesson_depend_exam_review) > 0 else ""
        #次のターン、スキルカードを引く 
        # 2ターン 後、スキルカードを引く
        if sum(self.exam_effect_timer_draw) > 0:
            if self.exam_effect_timer_draw[0] >= 1:
                str_ += "次のターン、スキルカードを{0}回引く\n".format(self.exam_effect_timer_draw[2])
            if self.exam_effect_timer_draw[0] > 1:
                str_ += "{0}ターン後、スキルカードを{1}回引く\n".format(self.exam_effect_timer_draw[0], self.exam_effect_timer_draw[2])
        #e_effect-exam_effect_timer-0001-01-e_effect-exam_lesson-0038-01
        # 次のターン、　パラメータ　を　38 上昇 1回
        if sum(self.exam_effect_timer_lesson) > 0:
            if self.exam_effect_timer_lesson[0] >= 1:
                str_ += "次のターン、パラメータを{0}上昇{1}回\n".format(self.exam_effect_timer_lesson[2], self.exam_effect_timer_lesson[3])
            if self.exam_effect_timer_lesson[0] > 1:
                str_ += "{0}ターン後、パラメータを{1}上昇{2}回\n".format(self.exam_effect_timer_lesson[0], self.exam_effect_timer_lesson[2], self.exam_effect_timer_lesson[3])
        # exam_effect_timer_card_upgrade = [0, 0]
        if sum(self.exam_effect_timer_card_upgrade) > 0:
            if self.exam_effect_timer_card_upgrade[0] >= 1:
                str_ += "次のターン、手札をすべて レッスン中強化\n"
            if self.exam_effect_timer_card_upgrade[0] > 1:
                str_ += "{0}ターン後、手札をすべて レッスン中強化\n".format(self.exam_effect_timer_card_upgrade[0])

        str_ += "追加ターン\n" if sum(self.exam_extra_turn) > 0 else ""
        str_ += "出牌数 + " + str(self.exam_playable_value_add[0]) + "\n" if sum(self.exam_playable_value_add) > 0 else ""
        str_ += "手札から墓地に移動し、同じ枚数のカードを引く\n" if sum(self.exam_hand_grave_count_card_draw) > 0 else ""
        str_ += "カードを" + str(self.exam_card_draw[0]) + "枚引く\n" if sum(self.exam_card_draw) > 0 else ""
        str_ += "ランダムな強化済みスキルカードを、手札に 生成\n" if sum(self.exam_card_create_search) > 0 else ""
        str_ += "負の効果無効" + str(self.exam_anti_debuff[0]) + "ターン\n" if sum(self.exam_anti_debuff) > 0 else ""
        str_ += "元気増加無効" + str(self.exam_block_restriction[0]) + "ターン\n" if sum(self.exam_block_restriction) > 0 else ""
        str_ += "次のカード効果を2回発動\n" if sum(self.exam_search_effect_play_count_buff) > 0 else ""
        str_ += "全ての手札をアップグレード\n" if sum(self.exam_card_upgrade) > 0 else ""
        # e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_034-enc01 以降、 アクティブスキルカード 使用時、 固定元気  +2 
        # e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_038-enc01 以降、 アクティブスキルカード 使用時、 集中  +1 
        # e_effect-exam_status_enchant-inf-enchant-p_card-01-men-3_035-enc01 以降、ターン終了時 集中 が3以上の場合、 集中  +2 
        # e_effect-exam_status_enchant-inf-enchant-p_card-02-ido-1_016-enc01 以降、ターン終了時、 好印象  +1 
        # e_effect-exam_status_enchant-inf-enchant-p_card-02-ido-3_019-enc01 以降、ターン終了時、 好印象  +1 
        # e_effect-exam_status_enchant-inf-enchant-p_card-02-men-2_004-enc01 以降、 メンタルスキルカード 使用時、 やる気  +1
        # e_effect-exam_status_enchant-inf-enchant-p_card-02-men-2_058-enc01 以降、 メンタルスキルカード 使用時、 好印象  +1 
        # e_effect-exam_status_enchant-inf-enchant-p_card-02-men-3_042-enc01 以降、ターン終了時 好印象 が3以上の場合、 好印象  +3 
        str_ += "以降、 アクティブスキルカード 使用時、 固定元気  +2\n" if self.exam_status_enchant[0] == 1 else ""
        str_ += "以降、 アクティブスキルカード 使用時、 集中  +1\n" if self.exam_status_enchant[0] == 2 else ""
        str_ += "以降、ターン終了時 集中 が3以上の場合、 集中  +2\n" if self.exam_status_enchant[0] == 3 else ""
        str_ += "以降、ターン終了時、 好印象  +1\n" if self.exam_status_enchant[0] == 4 else ""
        str_ += "以降、ターン終了時、 好印象  +1\n" if self.exam_status_enchant[0] == 5 else ""
        str_ += "以降、 メンタルスキルカード 使用時、 やる気  +1\n" if self.exam_status_enchant[0] == 6 else ""
        str_ += "以降、 メンタルスキルカード 使用時、 好印象  +1\n" if self.exam_status_enchant[0] == 7 else ""
        str_ += "以降、ターン終了時 好印象 が3以上の場合、 好印象  +3\n" if self.exam_status_enchant[0] == 8 else ""


        return str_
    
    def __repr__(self):
        return self.__str__()
    


# 1. 元气增加
def effect_exam_block(effect: Effect, game: Any):
    if game.block_restriction == 0 and effect.exam_block[0] > 0:
        game.block += effect.exam_block[0] + game.card_play_aggressive

# 2. やる気增加
def effect_exam_card_play_aggressive(effect: Effect, game: Any):
    game.card_play_aggressive += effect.exam_card_play_aggressive[0]

# 3. 加分
def effect_exam_lesson(effect: Effect, game: Any):
    for _ in range(effect.exam_lesson[1]):
        num = effect.exam_lesson[0]
        if game.lesson_buff > 0:
            num += game.lesson_buff
        mul = 1.0
        if game.parameter_buff > 0:
            mul += 0.5
        if game.parameter_buff_multiple_per_turn > 0:
            mul += game.parameter_buff * 0.1
        num = math.ceil(num * mul)
        game.lesson += num
    game.lesson = min(game.lesson, game.target_lesson)

# 4. 好印象增加
def effect_exam_review(effect: Effect, game: Any):
    game.review += effect.exam_review[0]

def effect_exam_depend(num, percent, time, game):
    #print(num, percent, time, game.lesson, game.target_lesson)
    for i in range(time):
        num = math.ceil(num * (percent/1000))
        game.lesson += num
    game.lesson = min(game.lesson, game.target_lesson)

# 5. 依赖元气加分
def effect_exam_lesson_depend_block(effect: Effect, game: Any):
    effect_exam_depend(game.block, effect.exam_lesson_depend_block[0], effect.exam_lesson_depend_block[2], game)
    game.block = math.ceil(game.block * (1 - effect.exam_lesson_depend_block[1] / 1000))
    
# 7. 体力消耗增加
def effect_exam_stamina_consumption_add(effect: Effect, game: Any):
    if effect.exam_stamina_consumption_add[0] == 0:
        return
    if game.stamina_consumption_add == 0:
        game.stamina_consumption_add_flag = True
    game.stamina_consumption_add += effect.exam_stamina_consumption_add[0]

# 8. 体力消耗减少
def effect_exam_stamina_consumption_down(effect: Effect, game: Any):
    if effect.exam_stamina_consumption_down[0] == 0:
        return
    if game.stamina_consumption_down == 0:
        game.stamina_consumption_down_flag = True
    game.stamina_consumption_down += effect.exam_stamina_consumption_down[0]

# 9. 直接体力消耗减少
def effect_exam_stamina_consumption_down_fix(effect: Effect, game: Any):
    game.stamina_consumption_down_fix += effect.exam_stamina_consumption_down_fix[0]

def effect_exam_stamina_recover_fix_once(effect: Effect, game: Any):
    game.stamina += effect.exam_stamina_recover_fix_once[0]
    game.stamina = min(game.stamina, game.max_stamina)
# 好调
def effect_exam_parameter_buff(effect: Effect, game: Any):
    if game.parameter_buff == 0:
        game.parameter_buff_flag = True
    game.parameter_buff += effect.exam_parameter_buff[0]
# 集中
def effect_exam_lesson_buff(effect: Effect, game: Any):
    game.lesson_buff += effect.exam_lesson_buff[0]

# 集中倍率增加的加分
def effect_exam_multiple_lesson_buff_lesson(effect: Effect, game: Any):
    for _ in range(effect.exam_multiple_lesson_buff_lesson[2]):
        num = effect.exam_multiple_lesson_buff_lesson[0]
        if game.lesson_buff > 0:
            num += (game.lesson_buff) * (effect.exam_multiple_lesson_buff_lesson[1] / 1000)
        mul = 1.0
        if game.parameter_buff > 0:
            mul += 0.5
        if game.parameter_buff_multiple_per_turn > 0:
            mul += game.parameter_buff * 0.1
        num = math.ceil(num * mul)
        game.lesson += num
    game.lesson = min(game.lesson, game.target_lesson)
def effect_exam_lesson_depend_exam_card_play_aggressive(effect: Effect, game: Any):
    effect_exam_depend(game.card_play_aggressive, effect.exam_lesson_depend_exam_card_play_aggressive[0], effect.exam_lesson_depend_exam_card_play_aggressive[1], game)
# 绝好调回合数
def effect_exam_parameter_buff_multiple_per_turn(effect: Effect, game: Any):
    if sum(effect.exam_parameter_buff_multiple_per_turn) == 0:
        return
    if game.parameter_buff_multiple_per_turn == 0:
        game.parameter_buff_multiple_per_turn_flag = True
    game.parameter_buff_multiple_per_turn += effect.exam_parameter_buff_multiple_per_turn[0]

def effect_exam_lesson_depend_exam_review(effect: Effect, game: Any):
    effect_exam_depend(game.review, effect.exam_lesson_depend_exam_review[0], effect.exam_lesson_depend_exam_review[1], game)

def effect_exam_effect_timer_draw(effect: Effect, game: Any):
    # e_effect-exam_effect_timer-(\d+)-(\d+)-e_effect-exam_card_draw-(\d+) -> 效果回合数，发动次数，抽卡次数
    for i in range(effect.exam_effect_timer_draw[0]):
        game.timer_draw[i] += effect.exam_effect_timer_draw[2] * effect.exam_effect_timer_draw[1]

def effect_exam_effect_timer_lesson(effect: Effect, game: Any):
    # e_effect-exam_effect_timer-(\d+)-(\d+)-e_effect-exam_lesson-(\d+)-(\d+) -> 效果回合数，发动次数，加分，加分次数
    # 保证只有1回合，次数只有1，加分次数只有1
    game.timer_lesson += effect.exam_effect_timer_lesson[2]

def effect_exam_effect_timer_card_upgrade(effect: Effect, game: Any):
    for i in range(effect.exam_effect_timer_card_upgrade[0]):
        game.timer_card_upgrade[i] += 1

def effect_exam_extra_turn(effect: Effect, game: Any):
    game.turn_left += effect.exam_extra_turn[0]

def effect_exam_playable_value_add(effect: Effect, game: Any):
    game.playable_value += effect.exam_playable_value_add[0]

def effect_exam_hand_grave_count_card_draw(effect: Effect, game: Any):
    if sum(effect.exam_hand_grave_count_card_draw) == 0:
        return
    game.hand_grave_count_card_draw = True

def effect_exam_card_draw(effect: Effect, game: Any):
    game.card_draw += effect.exam_card_draw[0]

def effect_exam_card_create_search(effect: Effect, game: Any):
    if sum(effect.exam_card_create_search) == 0:
        return
    game.card_create_search = True
    # 还未实现

def effect_exam_anti_debuff(effect: Effect, game: Any):
    if effect.exam_anti_debuff[0] == 0:
        return
    if game.anti_debuff == 0:
        game.anti_debuff_flag = True
    game.anti_debuff += effect.exam_anti_debuff[0]

def effect_exam_block_restriction(effect: Effect, game: Any):
    if effect.exam_block_restriction[0] == 0:
        return
    if game.block_restriction == 0:
        game.block_restriction_flag = True
    game.block_restriction += effect.exam_block_restriction[0]

def effect_exam_search_effect_play_count_buff(effect: Effect, game: Any):
    if sum(effect.exam_search_effect_play_count_buff) == 0:
        return
    game.search_effect_play_count_buff = True

def effect_exam_card_upgrade(effect: Effect, game: Any):
    if sum(effect.exam_card_upgrade) == 0:
        return
    game.card_upgrade = True

def effect_exam_status_enchant(effect: Effect, game: Any):
    # 转化为one-hot
    tmp = [0] * 8
    if sum(effect.exam_status_enchant) == 0:
        return
    tmp[effect.exam_status_enchant[0] - 1] += 1
    game.status_enchant = tmp


def effect_roll(effects: list[Effect], game):
    pass
    for effect in effects:
        if not triggers.check_trigger_exam_card(effect.card_play_trigger, game):
            continue
        #print(effect)
        effect_exam_block(effect, game)
        effect_exam_card_play_aggressive(effect, game)
        effect_exam_lesson(effect, game)
        effect_exam_review(effect, game)
        effect_exam_lesson_depend_block(effect, game)
        effect_exam_stamina_consumption_add(effect, game)
        effect_exam_stamina_consumption_down(effect, game)
        effect_exam_stamina_consumption_down_fix(effect, game)
        effect_exam_stamina_recover_fix_once(effect, game)
        effect_exam_parameter_buff(effect, game)
        effect_exam_lesson_buff(effect, game)
        effect_exam_multiple_lesson_buff_lesson(effect, game)
        effect_exam_lesson_depend_exam_card_play_aggressive(effect, game)
        effect_exam_parameter_buff_multiple_per_turn(effect, game)
        effect_exam_lesson_depend_exam_review(effect, game)
        effect_exam_effect_timer_draw(effect, game)
        effect_exam_effect_timer_lesson(effect, game)
        effect_exam_effect_timer_card_upgrade(effect, game)
        effect_exam_extra_turn(effect, game)
        effect_exam_playable_value_add(effect, game)
        effect_exam_hand_grave_count_card_draw(effect, game)
        effect_exam_card_draw(effect, game)
        effect_exam_card_create_search(effect, game)
        effect_exam_anti_debuff(effect, game)
        effect_exam_block_restriction(effect, game)
        effect_exam_search_effect_play_count_buff(effect, game)
        effect_exam_card_upgrade(effect, game)
        effect_exam_status_enchant(effect, game)
        #print(game)


def read_effect_of_card(card: dict):
    effect_list = card.get("playEffects")
    all_effects = []
    for effect in effect_list:
        #print(effect)
        effect_str = effect.get("produceExamEffectId")
        trigger_str = effect.get("produceExamTriggerId")
        effect_list = match_all_effects(effect_str)
        trigger_list = triggers.match_all_triggers_exam_card(trigger_str)
        tmp_effect = Effect()
        tmp_effect.set_effect(effect_list, trigger_list)
        all_effects.append(tmp_effect)
    return all_effects


if __name__ == "__main__":
    import yaml
    data = yaml.load(open("./yaml/ProduceCard.yaml", 'r',encoding='utf-8'), Loader=yaml.FullLoader)
    for card in data:
        print(card.get("name"))
        all_effects = read_effect_of_card(card)
        for effect in all_effects:
            obs = effect.observe()
            assert sum(obs) > 0, card.get("name")
            print(effect)


