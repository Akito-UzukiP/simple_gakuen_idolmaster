from stable_baselines3 import PPO, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.callbacks import CheckpointCallback
from utils.env import GakuenIdolMasterEnv
from utils.agent import CustomExtractor

policy_kwargs = dict(
    features_extractor_class=CustomExtractor,
    features_extractor_kwargs=dict(features_dim=128))

env = make_vec_env(GakuenIdolMasterEnv, n_envs=40)

# model = PPO(
#     "MultiInputPolicy",
#     env,
#     verbose=1,
#     tensorboard_log="./log",
#     device=get_device(),
#     policy_kwargs=policy_kwargs,
#     n_epochs=10,  # 保持训练轮数合理
#     n_steps=2048,  # 设置n_steps
#     batch_size=8192  # 调整批量大小，使其成为 n_steps * n_envs 的因数
# )
model = DQN(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log="./log",
    device=get_device(),
    policy_kwargs=policy_kwargs,
    batch_size=1024,  # 调整批量大小，使其成为 n_steps * n_envs 的因数
    exploration_final_eps=0.05,
)

# model = DQN.load("dqn_gakuen_idol_master", device=get_device(), env=env, force_reset=True)
# model.exploration_fraction = 0
# model.exploration_initial_eps = 0.05
# model.exploration_final_eps = 0.05
# model.exploration_final_eps = 0.05
# model.learning_starts = 0
# model.start_time = 0


# 设置保存检查点的回调函数，每保存10000步
# checkpoint_callback = CheckpointCallback(save_freq=81920, save_path='./models/',
#                                          name_prefix='dqn_gakuen_idol_master')

# 开始训练并使用回调函数保存检查点
if __name__ == "__main__":
    model.learn(total_timesteps=4096000, log_interval=1, reset_num_timesteps=False)

    # 手动保存模型（可选）
    model.save("dqn_gakuen_idol_master")
