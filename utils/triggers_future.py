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


def check_trigger_exam_card(trigger: list[int], game: Any):
    '''
    检查卡牌的触发器
    默认为True，对于每一个不为0的触发器位，若其对应的触发器条件不满足，则返回False
    '''
    assert len(trigger) == len(card_play_trigger)
    triggered = True
    if trigger[0]:
        if game.card_play_aggressive < 3:
            triggered = False
    if trigger[1]:
        if game.card_play_aggressive < 6:
            triggered = False
    if trigger[2]:
        if game.lesson_buff < 3:
            triggered = False
    if trigger[3]:
        if game.lesson_buff < 6:
            triggered = False
    if trigger[4]:
        if game.parameter_buff == 0:
            triggered = False
    if trigger[5]:
        if game.review < 1:
            triggered = False
    if trigger[6]:
        if game.review < 3:
            triggered = False
    if trigger[7]:
        if game.max_stamina * 0.5 > game.stamina:
            triggered = False
    return triggered
        
def check_trigger_start_turn(trigger: list[int], game: Any):
    '''
    检查回合开始的触发器
    默认为True，对于每一个不为0的触发器位，若其对应的触发器条件不满足，则返回False
    '''
    assert len(trigger) == len(start_turn_trigger)
    triggered = True
    if trigger[0]:
        if game.lesson / game.target_lesson > 1:
            triggered = False
    if trigger[1]:
        if game.block > 0:
            triggered = False
    if trigger[2]:
        if game.parameter_buff == 0:
            triggered = False
    if trigger[3]:
        if game.max_stamina * 0.5 > game.stamina:
            triggered = False
    if trigger[4]:
        if game.turn_progress < 3:
            triggered = False
    return triggered
