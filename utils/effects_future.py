import re
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
exam_lesson_depend_block = re.compile(r"e_effect-exam_lesson_depend_block-(\d+)-(\d+)-(\d+)$")
def match_exam_lesson_depend_block(string):
    default_values = [0, 0]
    if exam_lesson_depend_block.match(string):
        return [int(i) for i in exam_lesson_depend_block.match(string).groups(default_values)]
    return default_values
# e_effect-exam_lesson_depend_block-(\d+)-(\d+)-(\d+)$ -> 依赖元气千分比加分，加分次数
exam_lesson_depend_block = re.compile(r"e_effect-exam_lesson_depend_block-(\d+)-(\d+)$")
def match_exam_lesson_depend_block(string):
    default_values = [0, 0]
    if exam_lesson_depend_block.match(string):
        return [int(i) for i in exam_lesson_depend_block.match(string).groups(default_values)]
    return default_values
# e_effect-exam_lesson_depend_review-(\d+)-(\d+) -> 依赖好印象加分，加分次数
exam_lesson_depend_review = re.compile(r"e_effect-exam_lesson_depend_review-(\d+)-(\d+)$")
def match_exam_lesson_depend_review(string):
    default_values = [0, 0]
    if exam_lesson_depend_review.match(string):
        return [int(i) for i in exam_lesson_depend_review.match(string).groups(default_values)]
    return default_values
# e_effect-exam_lesson_depend_aggressive-(\d+)-(\d+) -> 依赖やる気加分，加分次数
exam_lesson_depend_aggressive = re.compile(r"e_effect-exam_lesson_depend_aggressive-(\d+)-(\d+)$")
def match_exam_lesson_depend_aggressive(string):
    default_values = [0, 0]
    if exam_lesson_depend_aggressive.match(string):
        return [int(i) for i in exam_lesson_depend_aggressive.match(string).groups(default_values)]
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
exam_effect_timer = re.compile(r"e_effect-exam_effect_timer-(\d+)-(\d+)-e_effect-exam_card_draw-(\d+)$")
def match_exam_effect_timer_draw(string):
    default_values = [0, 0, 0]
    if exam_effect_timer.match(string):
        return [int(i) for i in exam_effect_timer.match(string).groups(default_values)]
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
# e_effect-exam_card_create_search-0001-p_card_search-random-random_pool-p_random_pool-all-upgrade_1-1-hand-random-1_1 -> 上手一张SSR+
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
# e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_034-enc01 以降、 アクティブスキルカード 使用時、 固定元気  +2 
# e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_038-enc01 以降、 アクティブスキルカード 使用時、 集中  +1 
# e_effect-exam_status_enchant-inf-enchant-p_card-01-men-3_035-enc01 以降、ターン終了時 集中 が3以上の場合、 集中  +2 
# e_effect-exam_status_enchant-inf-enchant-p_card-02-ido-1_016-enc01 以降、ターン終了時、 好印象  +1 
# e_effect-exam_status_enchant-inf-enchant-p_card-02-ido-3_019-enc01 以降、ターン終了時、 好印象  +1 
# e_effect-exam_status_enchant-inf-enchant-p_card-02-men-2_004-enc01 以降、 メンタルスキルカード 使用時、 やる気  +1
# e_effect-exam_status_enchant-inf-enchant-p_card-02-men-2_058-enc01 以降、 メンタルスキルカード 使用時、 好印象  +1 
# e_effect-exam_status_enchant-inf-enchant-p_card-02-men-3_042-enc01 以降、ターン終了時 好印象 が3以上の場合、 好印象  +3 
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
           exam_lesson_depend_block, exam_lesson_depend_review, exam_lesson_depend_aggressive, 
           exam_stamina_consumption_add, exam_stamina_consumption_down, exam_stamina_consumption_down_fix, 
            exam_stamina_recover_fix_once, exam_parameter_buff, exam_lesson_buff, 
           exam_multiple_lesson_buff_lesson, exam_lesson_depend_exam_card_play_aggressive,exam_parameter_buff_multiple_per_turn, 
           exam_lesson_depend_exam_review, exam_effect_timer, exam_effect_timer_lesson, exam_effect_timer_card_upgrade,
           exam_extra_turn, exam_playable_value_add, exam_hand_grave_count_card_draw, exam_card_draw, exam_card_create_search,
           exam_anti_debuff, exam_block_restriction, exam_card_search_effect_play_count_buff, exam_card_upgrade] + [re.compile(i) for i in e_effects]


def match_all_effects(string):
    all_results = []

    # 调用所有的匹配函数并将结果连接起来
    all_results += match_exam_block(string) # 元气增加
    all_results += match_exam_card_play_aggressive(string) # やる気增加
    all_results += match_exam_lesson(string) # 加分
    all_results += match_exam_review(string) # 好印象增加
    all_results += match_exam_lesson_depend_block(string) # 依赖元气加分
    all_results += match_exam_lesson_depend_review(string) # 依赖好印象加分
    all_results += match_exam_lesson_depend_aggressive(string) # 依赖やる気加分
    all_results += match_exam_stamina_consumption_add(string) # 体力消耗增加
    all_results += match_exam_stamina_consumption_down(string) # 体力消耗减少
    all_results += match_exam_stamina_consumption_down_fix(string) # 固定体力消耗减少
    all_results += match_exam_stamina_recover_fix_once(string) # 固定回复体力，单次
    all_results += match_exam_parameter_buff(string) # 好调增加
    all_results += match_exam_lesson_buff(string) # 集中增加
    all_results += match_exam_multiple_lesson_buff_lesson(string) # 集中倍率增加
    all_results += match_exam_lesson_depend_exam_card_play_aggressive(string) # 依赖やる気加分
    all_results += match_exam_parameter_buff_multiple_per_turn(string) # 绝好调回合数
    all_results += match_exam_lesson_depend_exam_review(string) # 依赖好印象加分
    all_results += match_exam_effect_timer_draw(string) # 后续回合，抽卡
    all_results += match_exam_effect_timer_lesson(string) # 后续回合，加分
    all_results += match_exam_effect_timer_card_upgrade(string) # 后续回合，升级所有手牌
    all_results += match_exam_extra_turn(string) # 增加回合
    all_results += match_exam_playable_value_add(string) # 出牌数增加
    all_results += match_exam_hand_grave_count_card_draw(string) # 手牌去墓地，同样数量牌上手
    all_results += match_exam_card_draw(string) # 立刻抽卡
    all_results += match_exam_card_create_search(string) # 上手一张SSR+
    all_results += match_exam_anti_debuff(string) # 负面效果无效回合数
    all_results += match_exam_block_restriction(string) # 元气增加无效回合数
    all_results += match_exam_card_search_effect_play_count_buff(string) # 下一张卡效果发动2次
    all_results += match_exam_card_upgrade(string) # 升级所有手牌
    all_results += match_e_effects(string) # 特殊效果

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