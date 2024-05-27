from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
from env import GakuenIdolMasterEnv
from agent import CardTransformer, CustomExtractor


policy_kwargs = dict(
    features_extractor_class=CustomExtractor,
    features_extractor_kwargs=dict(features_dim=128)
)
    
env = make_vec_env(GakuenIdolMasterEnv, n_envs=5)
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./log",device=get_device(),policy_kwargs=policy_kwargs,n_epochs=20, batch_size=512)
#print(model.policy)
model.learn(total_timesteps=204800, log_interval=1)
# save
model.save("ppo_gakuen_idol_master")