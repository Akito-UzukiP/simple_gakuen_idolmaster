import numpy as np
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
    def __str__(self):
        name = self.name
        rarity = self.rarity
        stamina = self.stamina
        forceStamina = self.forceStamina
        costType = self.costType
        costValue =  self.costValue
        desc_str = self.descriptions
        return_text = f"计划类型: {plan_types[self.planType]}\n"
        return_text += f"类别: {category_types[self.category]}\n"
        return_text +=  f"{name}({rarity})\n体力: {stamina} 直接体力: {forceStamina}\n"
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
    for c in all_cards:
        if c.name == card.name + "+":
            return c
    return None


if __name__ == "__main__":
    import yaml
    all_cards = []
    with open("yaml/ProduceCard.yaml", "r", encoding='UTF-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    for card in data:
        all_cards.append(read_card(card))
    all_upgrade_cards = []
    for card in all_cards:
        upgrade_card = find_upgrade_card(card, all_cards)
        card.upgradeCard = upgrade_card
    for card in all_cards:
        print(card)
        print()

# id,upgradeCount,name,assetId,isCharacterAsset,rarity,planType,category,stamina,forceStamina,costType,costValue,playProduceExamTriggerId,playEffects,playMovePositionType,moveEffectTriggerType,moveProduceExamEffectIds,isEndTurnLost,isInitial,isRestrict,produceCardStatusEnchantId,searchTag,libraryHidden,noDeckDuplication,descriptions,unlockProducerLevel,rentalUnlockProducerLevel,evaluation,originIdolCardId,originSupportCardId,isInitialDeckProduceCard,effectGroupIds,viewStartTime,isLimited,order