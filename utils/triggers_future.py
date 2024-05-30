# e_trigger-exam_card_play-card_play_aggressive_up-3 -> 集中>3
# e_trigger-exam_card_play-card_play_aggressive_up-6 -> 集中>6
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
import re
card_play_trigger = {
    "e_trigger-exam_card_play-card_play_aggressive_up-3",
    "e_trigger-exam_card_play-card_play_aggressive_up-6",
    "e_trigger-exam_card_play-lesson_buff_up-3",
    "e_trigger-exam_card_play-lesson_buff_up-6",
    "e_trigger-exam_card_play-parameter_buff",
    "e_trigger-exam_card_play-review_up-1",
    "e_trigger-exam_card_play-review_up-3",
    "e_trigger-exam_card_play-stamina_up_multiple-500",
}
start_turn_trigger = {
    "e_trigger-exam_start_turn-condition_threshold_multiple_down-1000",
    "e_trigger-exam_start_turn-no_block",
    "e_trigger-exam_start_turn-parameter_buff",
    "e_trigger-exam_start_turn-stamina_up_multiple-500",
    "e_trigger-exam_start_turn-turn_progress_up-3"
}

def match_all_triggers_exam_card(text):
    '''
    返回one-hot编码的触发器
    '''
    trigger = [0] * len(card_play_trigger)
    for i, trig in enumerate(card_play_trigger):
        if re.search(trig, text):
            trigger[i] = 1
    return trigger
