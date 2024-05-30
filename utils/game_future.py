import random
import numpy as np
import math
from . import triggers_future, effects_future, cards_future
from .cards_future import Card
from .effects_future import Effect
# except:
#     import triggers_future, effects_future, cards_future
#     from cards_future import Card
#     from effects_future import Effect
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
        self.is_over = False
        self.plan = "ProducePlanType_Plan2" # ロジック
        self.max_stamina = 30
        self.stamina = 30
        self.block = 0
        self.card_play_aggressive = 0
        self.lesson = 0
        self.target_lesson = 60
        self.review = 0
        self.review_Flag = False
        # 每回合结束时减少的属性的Flag,在减少时判断如果为True，说明是第一次减少，不减少，将其置为False
        self.lesson_buff = 0
        self.parameter_buff = 0
        self.parameter_buff_Flag = False
        self.parameter_buff_multiple_per_turn = 0
        self.parameter_buff_multiple_per_turn_Flag = False

        self.card_draw = 3
        self.playable_value = 1

        self.stamina_consumption_add = 0
        self.stamina_consumption_add_Flag = False
        self.stamina_consumption_down = 0
        self.stamina_consumption_down_Flag = False
        self.stamina_consumption_down_fix = 0
        self.anti_debuff = 0
        self.anti_debuff_Flag = False
        self.block_restriction = 0
        self.block_restriction_Flag = False

        self.timer_card_draw = [0, 0]
        self.timer_lesson = 0
        self.timer_card_upgrade = [0, 0]
        self.search_effect_play_count_buff = 0
        self.status_enchant = [0] * 8



        self.hand_grave_count_card_draw = False # 在打出牌之后检查该Flag，如果为True，整体换牌
        self.card_create_search = False # 在打出牌后检查该Flag，如果为True，搜一张SSR+牌加入手牌
        self.search_effect_play_count_buff = False # 在打牌前检查该Flag，如果为True，下一张牌效果触发两次，不扣血 ### 可能存在上手两张国民アイドル导致该效果的对象仍然是该效果，过于稀有，暂时不考虑
        self.card_upgrade = False # 在打牌后检查该Flag，如果为True，手牌全部 * レッスン中強化 * 由于暂时没加入レッスン外強化，理解为不存在++卡就行


        # 手牌、牌库、弃牌堆、除外
        self.hand = []
        self.deck = []
        self.discard = []
        self.exile = []

        # 回合数、目标分数
        self.turn_left = 6
        self.target = 60
        self.current_turn = 0
        self.first_turn = True

    def __str__(self) -> str:
        str_ = "体力: " + str(self.stamina) + "/" + str(self.max_stamina) + "\n"
        str_ += "分数: " + str(self.lesson) + "/" + str(self.target_lesson) + "\n"
        str_ += "元気: " + str(self.block) + "\n" if self.block > 0 else ""
        str_ += "やる気: " + str(self.card_play_aggressive) + "\n" if self.card_play_aggressive > 0 else ""
        str_ += "好印象: " + str(self.review) + "\n" if self.review > 0 else ""
        str_ += "集中: " + str(self.lesson_buff) + "\n" if self.lesson_buff > 0 else ""
        str_ += "好调: " + str(self.parameter_buff) + "\n" if self.parameter_buff > 0 else ""
        str_ += "绝好调: " + str(self.parameter_buff_multiple_per_turn) + "\n" if self.parameter_buff_multiple_per_turn > 0 else ""

        # 计时器
        str_ += "体力消耗下降: " + str(self.stamina_consumption_down) + "回合\n" if self.stamina_consumption_down > 0 else ""
        str_ += "体力消耗增加: " + str(self.stamina_consumption_add) + "回合\n" if self.stamina_consumption_add > 0 else ""
        str_ += "体力消耗减少: " + str(self.stamina_consumption_down_fix) + "点\n" if self.stamina_consumption_down_fix > 0 else ""
        str_ += "抗debuff: " + str(self.anti_debuff) + "回合\n" if self.anti_debuff > 0 else ""
        str_ += "元气增长无效: " + str(self.block_restriction) + "回合\n" if self.block_restriction > 0 else ""
        str_ += "下一回合额外抽牌: " + str(self.timer_card_draw[0]) + "次，下下回合额外抽牌" + str(self.timer_card_draw[1]) + "次\n" if self.timer_card_draw[0] > 0 else ""
        str_ += "下一回合，额外分数: " + str(self.timer_lesson) + "分\n" if self.timer_lesson > 0 else ""
        str_ += "下一回合，所有手牌升级: " + str(self.timer_card_upgrade[0]) +( "张，下下回合" + str(self.timer_card_upgrade[1]) + "张\n" if self.timer_card_upgrade[1] > 0 else "") if self.timer_card_upgrade[0] > 0 else ""
        str_ += "下一张牌效果触发两次\n" if self.search_effect_play_count_buff else ""

        str_ += cards_future.print_cards(self.hand, card_per_line=3, max_symbols_per_line=50) if len(self.hand) > 0 else ""
        return str_
    
    

    def check_game_end(self):
        '''
        检查游戏是否结束
        '''
        if self.lesson >= self.target_lesson:
            return True
        if self.turn_left == 0:
            return True
        return False
    
    def status_enchant_card_play(self, card: Card):
        # e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_034-enc01 以降、 アクティブスキルカード 使用時、 固定元気  +2  0
        # e_effect-exam_status_enchant-inf-enchant-p_card-01-men-2_038-enc01 以降、 アクティブスキルカード 使用時、 集中  +1 1
        # e_effect-exam_status_enchant-inf-enchant-p_card-02-men-2_004-enc01 以降、 メンタルスキルカード 使用時、 やる気  +1 5
        # e_effect-exam_status_enchant-inf-enchant-p_card-02-men-2_058-enc01 以降、 メンタルスキルカード 使用時、 好印象  +1 6
        if card.category == "ProduceCardCategory_ActiveSkill":
            self.block += 2 * self.status_enchant[0]
            self.lesson_buff += 1 * self.status_enchant[1]
        if card.category == "ProduceCardCategory_MentalSkill":
            self.card_play_aggressive += 1 * self.status_enchant[5]
            self.review += 1 * self.status_enchant[6]

    def status_enchant_end_turn(self):
        # e_effect-exam_status_enchant-inf-enchant-p_card-01-men-3_035-enc01 以降、ターン終了時 集中 が3以上の場合、 集中  +2 2
        # e_effect-exam_status_enchant-inf-enchant-p_card-02-ido-1_016-enc01 以降、ターン終了時、 好印象  +1 3
        # e_effect-exam_status_enchant-inf-enchant-p_card-02-ido-3_019-enc01 以降、ターン終了時、 好印象  +1 4
        # e_effect-exam_status_enchant-inf-enchant-p_card-02-men-3_042-enc01 以降、ターン終了時 好印象 が3以上の場合、 好印象  +3 7
        if self.lesson_buff >= 3:
            self.lesson_buff += 2 * self.status_enchant[2]
        if self.review >= 1:
            self.review += 1 * self.status_enchant[3]
        if self.review >= 3:
            self.review += 1 * self.status_enchant[4]
        if self.review >= 3:
            self.review += 3 * self.status_enchant[7]


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


    def cost_stamina(self, cost):
        '''
        消耗体力，调用时应当保证体力和消耗体力的合法性
        '''
        if self.stamina_consumption_add:
            cost = math.ceil(cost * 2)
        if self.stamina_consumption_down:
            cost = math.ceil(cost / 2)
        if self.stamina_consumption_down_fix:
            cost -= self.stamina_consumption_down_fix
        cost = max(0, cost)
        if self.block > 0:
            if self.block >= cost:
                self.block -= cost
                return
            cost -= self.block
            self.block = 0
        self.stamina -= cost
    def cost_stamina_force(self, cost):
        '''
        直接消耗体力
        '''
        if self.stamina_consumption_add:
            cost = math.ceil(cost * 2)
        if self.stamina_consumption_down:
            cost = math.ceil(cost / 2)
        if self.stamina_consumption_down_fix:
            cost -= self.stamina_consumption_down_fix
        cost = max(0, cost)
        self.stamina -= cost

    def cost_special(self, cost_type, cost_value):
        '''
        消耗特殊资源，应当保证合法性
        '''
        # cost_classes = {
        #     "ExamCostType_ExamReview": "好印象",
        #     "ExamCostType_ExamCardPlayAggressive": "やる気",
        #     "ExamCostType_ExamLessonBuff": "集中",
        #     "ExamCostType_ExamParameterBuff": "好調",
        #     "ExamCostType_Unknown": "无"
        # }
        if cost_type == "ExamCostType_Unknown":
            return
        if cost_type == "ExamCostType_ExamReview":
            self.review -= cost_value
        elif cost_type == "ExamCostType_ExamCardPlayAggressive":
            self.card_play_aggressive -= cost_value
        elif cost_type == "ExamCostType_ExamLessonBuff":
            self.lesson_buff -= cost_value
        elif cost_type == "ExamCostType_ExamParameterBuff":
            self.parameter_buff -= cost_value

    def lesson_review(self):
        '''
        好印象加分
        '''
        self.lesson += self.review
        self.lesson = min(self.lesson, self.target_lesson)

    def check_playable(self, card_idx):
        '''
        检查打牌的合法性
        1. 体力消耗合法性
        2. 特殊资源消耗合法性
        3. 选择在手牌中的牌
        4. 检查trigger
        '''
        if card_idx >= len(self.hand) or card_idx < 0:
            return False
        card = self.hand[card_idx]
        
        cost_stamina = card.stamina
        force_stamina = card.forceStamina
        if self.stamina_consumption_add:
            cost_stamina = math.ceil(cost_stamina * 2)
            force_stamina = math.ceil(force_stamina * 2)
        if self.stamina_consumption_down:
            cost_stamina = math.ceil(cost_stamina / 2)
            force_stamina = math.ceil(force_stamina / 2)
        if self.stamina_consumption_down_fix:
            cost_stamina -= self.stamina_consumption_down_fix
            force_stamina -= self.stamina_consumption_down_fix
        cost_stamina = max(0, cost_stamina)
        force_stamina = max(0, force_stamina)
        if self.block + self.stamina < cost_stamina:
            return False
        if self.stamina < force_stamina:
            return False

        if triggers_future.check_trigger_start_turn(card.playableTrigger, self) == False:
            return False

        # 特殊资源消耗合法性
        if card.costType != "ExamCostType_Unknown":
            if card.costType == "ExamCostType_ExamReview":
                if self.review < card.costValue:
                    return False
            elif card.costType == "ExamCostType_ExamCardPlayAggressive":
                if self.card_play_aggressive < card.costValue:
                    return False
            elif card.costType == "ExamCostType_ExamLessonBuff":
                if self.lesson_buff < card.costValue:
                    return False
            elif card.costType == "ExamCostType_ExamParameterBuff":
                if self.parameter_buff < card.costValue:
                    return False

        return True


    def discard_card(self, card_idx):
        '''
        弃牌
        '''
        if self.hand[card_idx].playMovePositionType == "ProduceCardMovePositionType_Lost":
            self.exile.append(self.hand.pop(card_idx))
        else:
            self.discard.append(self.hand.pop(card_idx))


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

        self.playable_value = 1
        # if self.timer_lesson:
        #     effects_future.effect_exam_lesson(self.timer_lesson, self)

        # 虽然感觉不太可能，但是还是加上
        self.is_over = self.check_game_end()



        self.draw(self.card_draw)

        if self.timer_card_upgrade[0]:
            for i in range(len(self.hand)):
                self.hand[i] = self.hand[i].upgradeCard
            self.timer_card_upgrade[0] = self.timer_card_upgrade[1]
            self.timer_card_upgrade[1] = 0

        pass

    def rest(self):
        '''
        休息
        '''
        self.stamina += 2
        self.stamina = min(self.stamina, self.max_stamina)

    def play_card(self,card_idx):
        '''
        打出牌
        0. enchant结算
        1. 结算效果
        2. 结算是否结束回合

        这里不计算卡牌是否可以打出，应当在调用前检查
        '''
        assert card_idx < len(self.hand)
        card = self.hand[card_idx]
        self.status_enchant_card_play(card)
        effects = card.playEffects
        # 卡牌消耗结算
        self.cost_stamina(card.stamina)
        self.cost_stamina_force(card.forceStamina)
        self.cost_special(card.costType, card.costValue)
        # 卡牌效果结算
        if self.search_effect_play_count_buff:
            effects_future.effect_roll(effects, self)
            self.search_effect_play_count_buff = False
        effects_future.effect_roll(effects, self)

        
        # 抽牌、换牌、升级
        if self.hand_grave_count_card_draw:
            draw_num = len(self.hand)
            for _ in range(len(self.hand)):
                self.discard_card(0)
            self.draw(draw_num)
            self.hand_grave_count_card_draw = False
        if self.card_create_search:
            if self.plan == "ProducePlanType_Plan2":
                new_card = random.choice(cards_future.all_logic_upgraded_ssr_cards)
            else:
                new_card = random.choice(cards_future.all_sense_upgraded_ssr_cards)
            self.hand.append(new_card)
            self.card_create_search = False
        if self.card_upgrade:
            for i in range(len(self.hand)):
                self.hand[i] = self.hand[i].upgradeCard
            self.card_upgrade = False
        self.is_over = self.check_game_end()

        # 弃牌/除外
        self.discard_card(card_idx)
        self.playable_value -= 1
    
    def turn_process(self):
        '''
        回合进行
        直到不可出牌为止
        1. 打出牌
        2. 结算效果
        3. 结算是否结束回合
        '''
        while self.playable_value > 0:
            # pesudo input

            self.playable_value -= 1
        pass

    def end_turn(self):
        '''
        结束回合
        1. 弃牌
        2. 所有在回合结束时的效果结算（好印象加分）
        3. 所有在回合结束时减少的timer结算
        '''
        while len(self.hand) > 0:
            self.discard_card(0)

        self.turn_left -= 1
        self.current_turn += 1
        if self.turn_left == 0:
            self.is_over = True
        
        # enchant结算
        self.status_enchant_end_turn()

        # 好印象加分
        self.lesson_review()        

        # 结算timer
        if self.review:
            if self.review_Flag:
                self.review_Flag = False
            else:
                self.review -= 1
        if self.parameter_buff:
            if self.parameter_buff_Flag:
                self.parameter_buff_Flag = False
            else:
                self.parameter_buff -= 1
        
        if self.parameter_buff_multiple_per_turn:
            if self.parameter_buff_multiple_per_turn_Flag:
                self.parameter_buff_multiple_per_turn_Flag = False
            else:
                self.parameter_buff_multiple_per_turn -= 1

        if self.anti_debuff:
            if self.anti_debuff_Flag:
                self.anti_debuff_Flag = False
            else:
                self.anti_debuff -= 1
        
        if self.block_restriction:
            if self.block_restriction_Flag:
                self.block_restriction_Flag = False
            else:
                self.block_restriction -= 1
        if self.stamina_consumption_add:
            if self.stamina_consumption_add_Flag:
                self.stamina_consumption_add_Flag = False
            else:
                self.stamina_consumption_add -= 1
        if self.stamina_consumption_down:
            if self.stamina_consumption_down_Flag:
                self.stamina_consumption_down_Flag = False
            else:
                self.stamina_consumption_down -= 1

        self.is_over = self.check_game_end()

        pass


