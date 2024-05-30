from utils import game_future, cards_future, effects_future
game = game_future.Game()
deck = cards_future.all_logic_cards
game.deck = deck
game.start_turn()
print(game)
game.play_card(0)
print(game)
