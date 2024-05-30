import numpy as np
import random
try:
    from . import triggers_future, effects_future
except:
    import triggers_future, effects_future
category_types = {
    "ProduceCardCategory_MentalSkill": "支援",
    "ProduceCardCategory_ActiveSkill": "主动",
    "ProduceCardCategory_Trouble": "麻烦",
}
plan_types = {
    "ProducePlanType_Plan1": "センス",
    "ProducePlanType_Plan2": "ロジック",
    "ProducePlanType_Common": "共通",
}
cost_classes = {
    "ExamCostType_ExamReview": "好印象",
    "ExamCostType_ExamCardPlayAggressive": "やる気",
    "ExamCostType_ExamLessonBuff": "集中",
    "ExamCostType_ExamParameterBuff": "好調",
    "ExamCostType_Unknown": "无"
}

class Card:
    def __init__(self):
        self.id = ""
        self.name = ""
        self.rarity = ""
        self.stamina = 0
        self.forceStamina = 0
        self.costType = ""
        self.costValue = 0
        self.descriptions = []
        self.planType = ""
        self.category = ""


        self.noDeckDuplication = False
        self.isLimited = False
        self.isInitialDeckProduceCard = False
        self.isRestrict = False
        self.isInitial = False
        self.isEndTurnLost = False

        self.playMovePositionType = "ProduceCardMovePositionType_Grave" # 打出后该去墓地还是弃牌堆

        self.playableTrigger = [0] * len(triggers_future.start_turn_trigger)

        self.playEffects = []

        self.upgradeCard = None
        self.downgradeCard = None
    def __str__(self):
        name = self.name
        rarity = self.rarity
        stamina = self.stamina
        forceStamina = self.forceStamina
        costType = self.costType
        costValue =  self.costValue
        desc_str = self.descriptions
        # return_text = f"计划类型: {plan_types[self.planType]}\n"
        # return_text += f"类别: {category_types[self.category]}\n"
        return_text =  f"{name}({rarity})\n体力: {stamina} 直接体力: {forceStamina}\n"
        if self.isInitial:
            return_text += "レッスン 開始時手札に入る \n"
        if costType != "ExamCostType_Unknown":
            # ExamCostType_ExamReview -> 好印象
            # ExamCostType_ExamCardPlayAggressive -> やる気
            # ExamCostType_ExamLessonBuff -> 集中
            # ExamCostType_ExamParameterBuff -> 好調
            # ExamCostType_Unknown -> 无
            return_text += f"额外消耗: {costValue} {cost_classes[costType]}\n"
        for effect in self.playEffects:
            return_text += str(effect)
        if self.playMovePositionType == "ProduceCardMovePositionType_Lost":
            return_text += "レッスン中1回 "
        if self.noDeckDuplication:
            return_text += "重複不可 "
        return_text += "\n"

        
        return return_text

    def __repr__(self):
        return self.__str__()

def read_card(card_json:dict):
    card = Card()
    card.name = card_json.get('name')
    card.rarity = card_json.get('rarity').split("_")[-1]
    card.stamina = card_json.get('stamina')
    card.forceStamina = card_json.get('forceStamina')
    card.costType = card_json.get('costType')
    card.costValue = card_json.get('costValue')
    card.descriptions = card_json.get('descriptions')
    descriptions = card_json.get('descriptions')
    card.descriptions = ""
    for desc in descriptions:
        card.descriptions += desc.get('text').replace("<nobr>"," ").replace("</nobr>"," ").replace("\\n","") + " "
    card.planType = card_json.get('planType')
    card.category = card_json.get('category')

    card.noDeckDuplication = card_json.get('noDeckDuplication')
    card.isLimited = card_json.get('isLimited')
    card.isInitialDeckProduceCard = card_json.get('isInitialDeckProduceCard')
    card.isRestrict = card_json.get('isRestrict')
    card.isInitial = card_json.get('isInitial')
    card.isEndTurnLost = card_json.get('isEndTurnLost')
    card.playMovePositionType = card_json.get('playMovePositionType')

    card.playEffects = effects_future.read_effect_of_card(card_json)

    return card

def find_upgrade_card(card:Card, all_cards):
    # 还没加入支援卡，所以不考虑++的情况
    if card.name[-1] == "+":
        return card
    for c in all_cards:
        if c.name == card.name + "+":
            return c
    return None

import unicodedata

def get_display_width(text):
    """
    计算字符串的显示宽度，考虑全角和半角字符。
    """
    width = 0
    for char in text:
        if unicodedata.east_asian_width(char) in ('F', 'W', 'A'):
            width += 2
        else:
            width += 1
    return width

def print_cards(cards, card_per_line: int = 3, max_symbols_per_line: int = 50):
    """
    对于多张卡牌，漂亮地打印出来
    """
    card_strs = [str(card) for card in cards]
    card_strs_split = [card_str.split("\n") for card_str in card_strs]

    # Ensure each card is within max_symbols_per_line
    for idx, card_split in enumerate(card_strs_split):
        card_strs_split[idx] = [line[:max_symbols_per_line] for line in card_split]

    max_length = max([max([get_display_width(line) for line in card_str_split]) for card_str_split in card_strs_split])
    max_lines = max([len(card_str_split) for card_str_split in card_strs_split])

    output_str = "+" + ("-" * (max_length + 2) + "+") * card_per_line + "\n"
    printed_cards = 0

    while printed_cards < len(cards):
        for i in range(max_lines):
            output_str += "|"
            for j in range(card_per_line):
                card_idx = printed_cards + j
                if card_idx < len(cards):
                    if i < len(card_strs_split[card_idx]):
                        line = card_strs_split[card_idx][i]
                        display_width = get_display_width(line)
                        output_str += " " + line
                        output_str += " " * (max_length - display_width + 1)
                    else:
                        output_str += " " * (max_length + 2)
                else:
                    output_str += " " * (max_length + 2)
                output_str += "|"
            output_str += "\n"
        output_str += "+" + ("-" * (max_length + 2) + "+") * card_per_line + "\n"
        printed_cards += card_per_line

    return output_str


import yaml
all_cards = []
with open("yaml/ProduceCard.yaml", "r", encoding='UTF-8') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
for card in data:
    if "++" in card.get("name"):
        continue
    all_cards.append(read_card(card))
all_upgrade_cards = []
for card in all_cards:
    upgrade_card = find_upgrade_card(card, all_cards)
    card.upgradeCard = upgrade_card
    upgrade_card.downgradeCard = card
for card in all_cards:
    if card.downgradeCard is None:
        card.downgradeCard = card

all_logic_cards = [i for i in all_cards if i.planType == "ProducePlanType_Plan2"]
all_sense_cards = [i for i in all_cards if i.planType == "ProducePlanType_Plan1"]

def random_logic_card():
    return random.choice(all_logic_cards)
def random_sense_card():
    return random.choice(all_sense_cards)


if __name__ == "__main__":
    print(print_cards(all_logic_cards[:5], 3))

# id,upgradeCount,name,assetId,isCharacterAsset,rarity,planType,category,stamina,forceStamina,costType,costValue,playProduceExamTriggerId,playEffects,playMovePositionType,moveEffectTriggerType,moveProduceExamEffectIds,isEndTurnLost,isInitial,isRestrict,produceCardStatusEnchantId,searchTag,libraryHidden,noDeckDuplication,descriptions,unlockProducerLevel,rentalUnlockProducerLevel,evaluation,originIdolCardId,originSupportCardId,isInitialDeckProduceCard,effectGroupIds,viewStartTime,isLimited,order