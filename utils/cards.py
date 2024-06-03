import numpy as np
import random
try:
    from . import triggers, effects
except:
    import utils.triggers as triggers, utils.effects as effects
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

        self.playableTrigger = [0] * len(triggers.start_turn_trigger)

        self.playEffects = []

        self.upgradeCard = None
        self.downgradeCard = None
        self.obs_ = None

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
    
    def observe(self):
        if self.obs_:
            return self.obs_
        obs_ = []
        rarity = ["N", "R", "Sr", "Ssr"].index(self.rarity)
        obs_.append(rarity)
        planType = ["ProducePlanType_Plan1", "ProducePlanType_Plan2", "ProducePlanType_Common"].index(self.planType)
        obs_.append(planType)
        category = ["ProduceCardCategory_MentalSkill", "ProduceCardCategory_ActiveSkill", "ProduceCardCategory_Trouble"].index(self.category)
        obs_.append(category)
        obs_.append(self.stamina)
        obs_.append(self.forceStamina)
        obs_.append(self.costValue)
        obs_.append(["ExamCostType_ExamReview", "ExamCostType_ExamCardPlayAggressive", "ExamCostType_ExamLessonBuff", "ExamCostType_ExamParameterBuff", "ExamCostType_Unknown"].index(self.costType))
        obs_.append(1 if self.noDeckDuplication else 0)
        obs_.append(1 if self.isLimited else 0)
        obs_.append(1 if self.isInitialDeckProduceCard else 0)
        obs_.append(1 if self.isRestrict else 0)
        obs_.append(1 if self.isInitial else 0)
        obs_.append(1 if self.isEndTurnLost else 0)
        obs_.append(["ProduceCardMovePositionType_Grave", "ProduceCardMovePositionType_Lost"].index(self.playMovePositionType))
        obs_.append([i.observe() for i in self.playEffects])
        self.obs_ = obs_
        return obs_


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

    card.playEffects = effects.read_effect_of_card(card_json)

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
    if upgrade_card is not None:
        card.upgradeCard = upgrade_card
        upgrade_card.downgradeCard = card
    else:
        card.upgradeCard = card
        card.downgradeCard = card
for card in all_cards:
    if card.downgradeCard is None:
        card.downgradeCard = card

all_logic_cards = [i for i in all_cards if i.planType == "ProducePlanType_Plan2"]
all_sense_cards = [i for i in all_cards if i.planType == "ProducePlanType_Plan1"]
all_logic_upgraded_ssr_cards = [i for i in all_logic_cards if i.rarity == "Ssr" and i.upgradeCard == i]
all_sense_upgraded_ssr_cards = [i for i in all_sense_cards if i.rarity == "Ssr" and i.upgradeCard == i]
all_logic_upgraded_cards = [i for i in all_logic_cards if i.upgradeCard == i]
all_sense_upgraded_cards = [i for i in all_sense_cards if i.upgradeCard == i]
def random_logic_card():
    return random.choice(all_logic_cards)
def random_sense_card():
    return random.choice(all_sense_cards)

def search_card_by_name(name:str) -> Card:
    for card in all_cards:
        if card.name == name:
            return card
    return None
def sc(name:str) -> Card:
    return search_card_by_name(name)
def add_card_to_deck(deck, card_name, num):
    card = sc(card_name)
    if card is None:
        #print(f"没有找到卡牌{card_name}")
        return deck, False
    if card.noDeckDuplication:
        if card in deck or card.upgradeCard in deck or card.downgradeCard in deck:
            #print(f"卡牌{card_name}不能重复")
            return deck, False
        else:
            deck += [card] * 1
            return deck, True
    else:
        deck += [card] * num
        return deck, True

def create_deck(deck_dict):
    deck = []
    for card_name, num in deck_dict.items():
        deck, success = add_card_to_deck(deck, card_name, num)
        if not success:
            return None
    return deck 



kotone_deck_dict = {
    "アピールの基本": 2,
    "ポーズの基本": 1,
    "表現の基本": 2,
    "目線の基本": 2,
    "可愛い仕草": 1,
    "よそ見はダメ♪+": 1,
    "イメトレ": 1,
    "本番前夜+": 1,
    "やる気は満点": 3,
    "私がスター+": 1,
    "ふれあい": 3,
    "幸せな時間": 1,
    "手拍子+": 1,
}
kotone_deck = create_deck(kotone_deck_dict)
#print(print_cards(kotone_deck, 3, 50))
hiro_deck_dict = {
    "アピールの基本": 2,
    "ポーズの基本": 1,
    "表現の基本": 2,
    "意識の基本": 1,
    "気分転換": 1,
    "本気の趣味+": 1, # 固有
    "イメトレ": 1,  # 继承
    "本番前夜+": 1, # 继承
    # "本番前夜": 2,
    #"やる気は満点": 3,
    "私がスター+": 1, # 继承
    #"ふれあい": 1, 
    # "幸せな時間": 1,
    "元気な挨拶+": 1, # 继承
   # "パステル気分+": 1,
    # "あふれる思い出": 2,
    # "あふれる思い出+": 3,
    # "開花": 1, 
    # "気合十分！+": 1,
    # "えいえいおー": 2,
    # "叶えたい夢+": 1,
    # "光のステージ": 1,  
    # "ゆるふわおしゃべり+": 1,
}
hiro_full_deck_dict = {
    "アピールの基本": 2,
    "ポーズの基本": 1,
    "表現の基本": 2,
    "意識の基本": 1,
    "気分転換": 3,
    "本気の趣味+": 1, # 固有
    "イメトレ": 4,  # 继承
    "本番前夜+": 1, # 继承
    "本番前夜": 2,
    # "やる気は満点": 3,
    "私がスター+": 1, # 继承
    # "ふれあい": 1, 
    # "幸せな時間": 1,
    "元気な挨拶+": 1, # 继承
   "パステル気分+": 1,
    "あふれる思い出": 2,
    "あふれる思い出+": 3,
    "開花": 1, 
    "気合十分！+": 1,
    "えいえいおー": 2,
    "叶えたい夢+": 1,
    "光のステージ": 1,  
    "ハートの合図+": 2,
    # "ゆるふわおしゃべり+": 1,
}
temp_deck = create_deck({
    "私がスター+": 1,
    "イメトレ": 2,
    "幸せな時間": 1,
    "本気の趣味+": 1,
    "パステル気分+": 3,
})

all_block_related_cards = [
    "意識の基本",
    "えいえいおー",
    "あふれる思い出",
    "わくわくが止まらない",
    "本番前夜",
    "ひなたぼっこ",
    "イメトレ",
    "思い出し笑い",
    "パステル気分",
    "リズミカル",
    "励まし",
    # Active
    "気分転換",
    "ゆるふわおしゃべり",
    "元気な挨拶",
    "ありがとうの言葉",
    "ハートの合図",
    "キラメキ",
    "開花"
]
all_block_related_cards = [sc(i) for i in all_block_related_cards]
all_block_related_cards = [i for i in all_block_related_cards if i is not None]
all_block_related_cards += [i.upgradeCard for i in all_block_related_cards]

hiro_deck = create_deck(hiro_deck_dict)

def random_hiro_deck(additional_card = 15, plus_card = 5):
    deck = hiro_deck.copy()
    base_deck_length = len(deck)
    while len(deck) < base_deck_length + additional_card:
        deck, _ = add_card_to_deck(deck, random.choice(all_logic_cards).name, 1)
    for i in range(plus_card):
        deck, _ = add_card_to_deck(deck, random.choice(all_logic_upgraded_cards).name, 1)
    return deck

def random_hiro_block_deck(additional_card = 15):
    deck = hiro_deck.copy()
    base_deck_length = len(deck)
    while len(deck) < base_deck_length + additional_card:
        deck, _ = add_card_to_deck(deck, random.choice(all_block_related_cards).name, 1)
    return deck

if __name__ == "__main__":
    deck = random_hiro_block_deck()
    print(print_cards(deck, 3, 50))
# id,upgradeCount,name,assetId,isCharacterAsset,rarity,planType,category,stamina,forceStamina,costType,costValue,playProduceExamTriggerId,playEffects,playMovePositionType,moveEffectTriggerType,moveProduceExamEffectIds,isEndTurnLost,isInitial,isRestrict,produceCardStatusEnchantId,searchTag,libraryHidden,noDeckDuplication,descriptions,unlockProducerLevel,rentalUnlockProducerLevel,evaluation,originIdolCardId,originSupportCardId,isInitialDeckProduceCard,effectGroupIds,viewStartTime,isLimited,order

