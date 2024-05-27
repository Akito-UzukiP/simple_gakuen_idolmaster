# 游戏流程
import cards
from game import Game
game = Game(hp=30, total_turn=8, target=60)
game.deck = cards.create_ktn_deck()
game.shuffle()
while not game.is_over:
    game.start_round()
    print(game)
    # 等待输入
    action = -1
    while action < 0 or action >= len(game.hand):
        action = int(input("Please input the card index to play: "))
    game.play(action)
    print(game)
    game.end_round()
print("Game Over")