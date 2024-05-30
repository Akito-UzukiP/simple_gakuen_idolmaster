from utils import game_future, cards_future, effects_future
from utils.cards_future import Card, search_card_by_name as sc
game = game_future.Game()
kotone_deck = [sc("アピールの基本")] * 2 + [sc("ポーズの基本")] + [sc("表現の基本")] * 2 + [sc("目線の基本")] * 2 + [sc("可愛い仕草")] + [sc("よそ見はダメ♪+")]
game.deck = kotone_deck
while not game.is_over:
    game.start_turn()
    print(game)
    while game.playable_value > 0:
        action = 99
        while game.check_playable(action) == False:
            action = int(input("输入打出的牌(从0开始, -1表示休息[体力+2]): "))
            if action == -1:
                game.rest()
                break
        if not action == -1:
            game.play_card(action)
        print(game)
    game.end_turn()