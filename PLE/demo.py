from ple.games.flappybird import FlappyBird
from ple import PLE


game = FlappyBird()
p = PLE(game, fps=30, display_screen=True)

p.init()
reward = 0.0

for i in range(1000):
   if p.game_over():
           p.reset_game()

   observation = p.getScreenRGB()
   action = 1
   reward = p.act(action)