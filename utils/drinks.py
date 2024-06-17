# 饮料类
# 和打牌类似，仅在回合中使用，故相应效果判定与打牌相同
import numpy as np
import random
try:
    from . import effects
except:
    import utils.effects as effects

class Drinks:
    def __init__(self):
        self.id = ""
        self.name = ""
        self.rarity = ""
        self.descriptions = []
        self.playEffects = []
    def __str__(self):
        name = self.name
        rarity = self.rarity
        desc_str = self.descriptions
        # return_text = f"计划类型: {plan_types[self.planType]}\n"
        # return_text += f"类别: {category_types[self.category]}\n"
        return_text = f"饮料名称: {name}\n"
        for effect in self.playEffects:
            return_text += str(effect)
        return_text += "\n"

        
        return return_text

    def __repr__(self):
        return self.__str__()
    
    def observe(self):
        if self.obs_:
            return self.obs_
        obs_ = []
        obs_.append([i.observe() for i in self.playEffects])
        self.obs_ = obs_
        return obs_
    
def read_drink(drink_json: dict):
    drink = Drinks()
    drink.id = drink_json['id']
    drink.name = drink_json['name']
    drink.rarity = drink_json['rarity']
    drink.descriptions = drink_json['descriptions']
    drink.playEffects = effects.read_effect_of_drink(drink_json)
    return drink


import yaml, os
all_cards = []
if os.path.isfile('yaml/ProduceDrink.yaml'):
    filepath = 'yaml/ProduceDrink.yaml'
else:
    filepath = '../yaml/ProduceDrink.yaml'
with open(filepath, 'r', encoding='utf-8') as f:
    drinks = yaml.load(f, Loader=yaml.FullLoader)
    for drink in drinks:
        all_cards.append(read_drink(drink))
print(all_cards)