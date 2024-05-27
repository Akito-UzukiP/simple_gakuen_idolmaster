# 白伤
    # def __init__(self):
    #     self.hp = 30
    #     self.robust = 0
    #     self.good_impression = 0
    #     self.good_condition = 0
    #     self.best_condition = 0
    #     self.motivation = 0

def score(game, num):
    if game.good_condition > 0 and game.best_condition > 0:
        num += int(num * (1.5+0.1*game.good_condition))
    elif game.good_condition > 0:
        num += int(num * (1.5))
    game.score += num

def robust(game, num):
    game.robust += num + game.motivation
def good_impression(game, num):
    game.good_impression += num
def good_condition(game, num):
    game.good_condition += num
def best_condition(game, num):
    game.best_condition += num


def cost(game, num):
    # 先结算体力消耗增加，再结算体力消耗减少，再结算直接体力消耗减少
    if game.hp_damage_increase > 0:
        num *= 2
    if game.hp_damage_decrease > 0:
        num /= 2
    if game.hp_damage_decrease_direct > 0:
        num -= game.hp_damage_decrease_direct
    if game.robust > 0:
        if game.robust >= num:
            game.robust -= num
        else:
            game.hp -= (num - game.robust)
            game.robust = 0
    else:
        game.hp -= num
    game.hp = min(game.hp, game.init_hp)

def direct_cost(game, num):
    if game.hp_damage_increase > 0:
        num *= 2
    if game.hp_damage_decrease > 0:
        num /= 2
    if game.hp_damage_decrease_direct > 0:
        num -= game.hp_damage_decrease_direct
    game.hp -= num

def score_robust_percent(game, percent):
    tmp = int(game.robust * percent)
    score(game, tmp)
def score_good_impression_percent(game, percent):
    tmp = int(game.good_impression * percent)
    score(game, tmp)
def score_good_impression(game):
    score(game, game.good_impression)
def motivation(game, num):
    game.motivation += num

#体力消耗增加
def hp_damage_increase(game, num):
    game.hp_damage_increase += num
#体力消耗减少
def hp_damage_decrease(game, num):
    game.hp_damage_decrease += num
#直接体力消耗减少
def hp_damage_decrease_direct(game, num):
    game.hp_damage_decrease_direct += num
#额外抽牌
def additional_draw(game, num):
    game.draw(num)
#额外回合数
def additional_turn(game, num):
    game.turn_left += num
def additional_playable(game, num):
    game.playable_cnt += num


def end_turn(game):
    if game.good_condition > 0:
        game.good_condition -= 1
    if game.best_condition > 0:
        game.best_condition -= 1
    if game.good_impression > 0:
        game.good_impression -= 1

def effect_roll(effects: list[int], game):
    ''' 效果轮
    Input: 
        effects: 效果列表，数字表示效果类型
    Output:
        None
    Description:
        依次执行效果列表中的效果，按照顺序传入效果参数，执行效果，理论上传入的0应当表示无效果
        顺序为 cost, direct_cost, robust, good_impression, good_condition, best_condition, score, score_robust_percent, score_good_impression_percent,
        hp_damage_increase, hp_damage_decrease, hp_damage_decrease_direct, additional_playable, additional_draw, additional_turn
    '''
    #assert len(effects) == 10
    cost(game, effects[0])
    direct_cost(game, effects[1])
    robust(game, effects[2])
    motivation(game, effects[3])
    good_impression(game, effects[4])
    good_condition(game, effects[5])
    best_condition(game, effects[6])
    score(game, effects[7])
    score_robust_percent(game, effects[8])
    score_good_impression_percent(game, effects[9])
    hp_damage_increase(game, effects[10])
    hp_damage_decrease(game, effects[11])
    hp_damage_decrease_direct(game, effects[12])
    additional_playable(game, effects[13])
    additional_draw(game, effects[14])
    additional_turn(game, effects[15])
    score_good_impression(game)
    end_turn(game)
