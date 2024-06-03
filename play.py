from utils import cards, effects, game
from utils.cards import Card, search_card_by_name as sc
import random
game = game.Game()
game.deck = cards.hiro_deck
print(game.observe())
while not game.is_over:
    game.start_turn()
    while game.playable_value > 0:
        print(game)
        print(game.stamina_consumption_down_Flag)
        print(game.stamina_consumption_down)
        action = 99
        while game.check_playable(action) == False:
            action = int(input("输入打出的牌(从0开始, -1表示休息[体力+2]): "))
            print(game.check_playable(action))
            if action == -1:
                game.rest()
                break
        if not action == -1:
            game.play_card(action)
        #print(len(game.deck), len(game.hand), len(game.discard), len(game.exile))
    game.end_turn()