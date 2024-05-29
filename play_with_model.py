from stable_baselines3 import PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
from utils.env import GakuenIdolMasterEnv
from utils.agent import CardTransformer, CustomExtractor
from utils import agent
from utils import cards
import torch
def update_environment(env, action):
    obs, reward, done, info = env.step(action)
    return obs, reward, done, info
def predict_action(model, obs):
    #print(obs)
    with torch.no_grad():
        action, logprob = model.predict(obs, deterministic=True)
    return action

    
env = GakuenIdolMasterEnv()
model = DQN.load("models/dqn_gakuen_idol_master", device='cpu', env=env)

env.reset( )
print(env.game)
obs = env._get_obs()
done = False
total_rewards = 0
cnt = 0
while not done:
    # 使用模型进行预测
    action = predict_action(model, obs)

    if action >= len(env.game.hand):
        break
    print(env.game.hand[action])
    # 使用环境进行更新
    obs, reward, done, info = update_environment(env, action)
    
    total_rewards += reward
    print(env.game)
    cnt += 1
    if cnt > 30:
        break
print(f'Total reward: {total_rewards}')