# 游戏流程
import cards
from game import Game
game = Game(hp=1, total_turn=8, target=60)
game.deck = cards.create_ktn_deck()
game.shuffle()
while not game.is_over:
    game.start_round()
    # 等待输入
    print(game.observe())
    print(game)
    action = -1
    first = True
    while (action < 0 or action >= len(game.hand)) and (first or not game.check_playable(action)):
        action = int(input("Please input the card index to play: "))
        print(game.check_playable(action))
        first = False
    game.play(action)
    print(game)
    game.end_round()
print("Game Over")