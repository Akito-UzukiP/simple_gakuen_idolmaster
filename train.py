from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_device
from env import GakuenIdolMasterEnv
from agent import CardTransformer

class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CustomExtractor, self).__init__(observation_space, features_dim)
        self.card_transformer = CardTransformer(card_feature_dim=13, player_feature_dim=10,d_model=features_dim)
    def forward(self, observations):
        game = observations['game']
        card = observations['card']
        card = self.card_transformer(card, game)
        return card

policy_kwargs = dict(
    features_extractor_class=CustomExtractor,
    features_extractor_kwargs=dict(features_dim=128)
)
    
env = make_vec_env(GakuenIdolMasterEnv, n_envs=5)
model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./log",device=get_device(),policy_kwargs=policy_kwargs,n_epochs=10, batch_size=256)
#print(model.policy)
model.learn(total_timesteps=204800, log_interval=1,progress_bar=True)