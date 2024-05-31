from utils import agent_future, game_future, cards_future, effects_future, triggers_future, env_future
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3 import PPO, DQN
player_info_dim = 32
card_info_dim = 15
effect_info_dim = 49
max_cards = 40  # 假设每个玩家最多有10张卡片
max_effects_per_card = 4  # 假设每张卡片最多有5个效果
policy_kwargs = dict(
    features_extractor_class=agent_future.CustomExtractor,
    features_extractor_kwargs=dict(player_info_dim=player_info_dim, card_info_dim=card_info_dim, effect_info_dim=effect_info_dim, d_model=128))

env = make_vec_env(env_future.GakuenIdolMasterEnv, n_envs=10)

model = DQN(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./log",
    device=get_device(),
    policy_kwargs=policy_kwargs,
    batch_size=1024,  # 调整批量大小，使其成为 n_steps * n_envs 的因数
    exploration_final_eps=0.05,
    buffer_size = 1000000,
)

model.learn(total_timesteps=327680, log_interval=1)
model.save("dqn_gakuen_idol_master")