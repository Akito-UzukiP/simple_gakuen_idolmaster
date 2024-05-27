# 游戏流程
import cards
from game import Game
game = Game(hp=31, total_turn=8, target=60)
game.deck = cards.create_random_ktn_deck()
game.shuffle()
while not game.is_over:
    game.start_round()
    # 等待输入
    #print(game.observe())
    print(game)
    action = 99
    while game.check_playable(action) == False:
        action = int(input("Please input the card index to play: "))
        print(game)

    game.play(action)
    print(game)
    game.end_round()
print("Game Over")