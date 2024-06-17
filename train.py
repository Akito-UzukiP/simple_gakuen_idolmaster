from utils import agent, cards, effects, env, game, triggers
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3 import PPO, DQN, A2C, SAC

import tensorboard
player_info_dim = 32
card_info_dim = 15
effect_info_dim = 49
max_cards = 8  # 只观察手牌，最多5张
max_effects_per_card = 4  # 假设每张卡片最多有5个效果
policy_kwargs = dict(
    features_extractor_class=agent.CustomExtractor,
    features_extractor_kwargs=dict(player_info_dim=player_info_dim, card_info_dim=card_info_dim, effect_info_dim=effect_info_dim, d_model=64, max_cards=max_cards, max_effects_per_card=max_effects_per_card))

env = make_vec_env(env.GakuenIdolMasterEnv, n_envs=4, env_kwargs={'max_cards': 8})
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./log",
    device=get_device(),
    policy_kwargs=policy_kwargs,
    batch_size=4096,  # 调整批量大小，使其成为 n_steps * n_envs 的因数
    n_steps=1024,  # 一次训练的步数
    n_epochs=20
    
)
model = PPO.load("ppo_gakuen_idol_master_hiro_block", device=get_device(), env=env, force_reset=True)

model.learn(total_timesteps=3276800*2, log_interval=1,reset_num_timesteps=True, progress_bar=True)
model.save("ppo_gakuen_idol_master_hiro_block")