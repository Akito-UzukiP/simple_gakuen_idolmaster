# e_trigger-exam_card_play-card_play_aggressive_up-3 -> やる気>3
# e_trigger-exam_card_play-card_play_aggressive_up-6 -> やる気>6
# e_trigger-exam_card_play-lesson_buff_up-3 -> 集中>3
# e_trigger-exam_card_play-lesson_buff_up-6 -> 集中>6
# e_trigger-exam_card_play-parameter_buff -> 好调状态
# e_trigger-exam_card_play-review_up-1 -> 好印象>1
# e_trigger-exam_card_play-review_up-3 -> 好印象>3
# e_trigger-exam_card_play-stamina_up_multiple-500 -> 体力比例大于千分之500
# e_trigger-exam_start_turn-condition_threshold_multiple_down-1000 -> 分数比例小于千分之1000
# e_trigger-exam_start_turn-no_block
# e_trigger-exam_start_turn-parameter_buff
# e_trigger-exam_start_turn-stamina_up_multiple-500
# e_trigger-exam_start_turn-turn_progress_up-3
from typing import Any
import re
card_play_trigger = [
    "e_trigger-exam_card_play-card_play_aggressive_up-3",
    "e_trigger-exam_card_play-card_play_aggressive_up-6",
    "e_trigger-exam_card_play-lesson_buff_up-3",
    "e_trigger-exam_card_play-lesson_buff_up-6",
    "e_trigger-exam_card_play-parameter_buff",
    "e_trigger-exam_card_play-review_up-1",
    "e_trigger-exam_card_play-review_up-3",
    "e_trigger-exam_card_play-stamina_up_multiple-500",
]
start_turn_trigger = [
    "e_trigger-exam_start_turn-condition_threshold_multiple_down-1000",
    "e_trigger-exam_start_turn-no_block",
    "e_trigger-exam_start_turn-parameter_buff",
    "e_trigger-exam_start_turn-stamina_up_multiple-500",
    "e_trigger-exam_start_turn-turn_progress_up-3"
]

def match_all_triggers_exam_card(text):
    '''
    返回one-hot编码的触发器
    '''
    trigger = [0] * len(card_play_trigger)
    for i, trig in enumerate(card_play_trigger):
        if re.search(trig, text):
            trigger[i] = 1
    return trigger

def match_all_triggers_start_turn(text):
    '''
    返回one-hot编码的触发器
    '''
    trigger = [0] * len(start_turn_trigger)
    for i, trig in enumerate(start_turn_trigger):
        if re.search(trig, text):
            trigger[i] = 1
    return trigger


triggers_exam_card = {
    0: lambda g: g.card_play_aggressive >= 3,
    1: lambda g: g.card_play_aggressive >= 6,
    2: lambda g: g.lesson_buff >= 3,
    3: lambda g: g.lesson_buff >= 6,
    4: lambda g: g.parameter_buff > 0,
    5: lambda g: g.review >= 1,
    6: lambda g: g.review >= 3,
    7: lambda g: g.max_stamina * 0.5 <= g.stamina,
}

def check_trigger_exam_card(trigger: list[int], game: Any):
    '''
    检查卡牌的触发器
    默认为True，对于每一个不为0的触发器位，若其对应的触发器条件不满足，则返回False
    '''
    assert len(trigger) == len(card_play_trigger)
    triggered = True
    for i, t in enumerate(trigger):
        # 如果触发器位为1，但触发器条件不满足，则返回False
        if t and not triggers_exam_card[i](game):
            triggered = False
            break
    return triggered



triggers_start_turn = {
    0: lambda g: g.lesson / g.target_lesson <= 1,
    1: lambda g: g.block == 0,
    2: lambda g: g.parameter_buff != 0,
    3: lambda g: g.max_stamina * 0.5 < g.stamina,
    4: lambda g: g.turn_progress >= 3
}

def check_trigger_start_turn(trigger: list[int], game: Any):
    '''
    检查回合开始的触发器
    默认为True，对于每一个不为0的触发器位，若其对应的触发器条件不满足，则返回False
    '''
    assert len(trigger) == len(start_turn_trigger)
    triggered = True
    for i, t in enumerate(trigger):
        # 如果触发器位为1，但触发器条件不满足，则返回False
        if t and not triggers_start_turn[i](game):
            triggered = False
            break
    return triggered
        
