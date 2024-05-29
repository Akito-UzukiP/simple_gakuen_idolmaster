import numpy as np
try:
    from .effects_future import match_all_effects
except:
    from effects_future import match_all_effects
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
        self.name = ""
        self.rarity = ""
        self.stamina = 0
        self.forceStamina = 0
        self.costType = ""
        self.costValue = 0
        self.descriptions = []
        self.planType = ""
        self.category = ""

        self.playEffects = [0]*45
        self.trigger = [0]*8 # 8种trigger
        self.specialEffects = [0]*45 # trigger后的效果
        self.needTriggerToPlay = False

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
        return_text +=  f"{name}({rarity})\n{desc_str}\n体力: {stamina}+{forceStamina}\n"
        if costType != "ExamCostType_Unknown":
            # ExamCostType_ExamReview -> 好印象
            # ExamCostType_ExamCardPlayAggressive -> やる気
            # ExamCostType_ExamLessonBuff -> 集中
            # ExamCostType_ExamParameterBuff -> 好調
            # ExamCostType_Unknown -> 无
            return_text += f"额外消耗: {costValue} {cost_classes[costType]}\n"
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
    card.playEffects = [0]*45
    card.specialEffects = [0]*45
    for effect in card_json.get('playEffects'):
        if effect.get("produceExamTriggerId"):
            card.specialEffects = [a + b for a, b in zip(card.specialEffects, match_all_effects(effect.get("produceExamEffectId")))]
        elif effect.get("produceExamEffectId"):
            card.playEffects = [a + b for a, b in zip(card.playEffects, match_all_effects(effect.get("produceExamEffectId")))]
    return card



if __name__ == "__main__":
    import yaml
    all_cards = []
    with open("yaml/ProduceCard.yaml", "r", encoding='UTF-8') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    for card in data:
        all_cards.append(read_card(card))
    print(all_cards)