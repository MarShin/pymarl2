from typing import Any
import torch as th
from components.episode_buffer import EpisodeBatch
from controllers.basic_controller import BasicMAC
from modules.mixers.qmix import QMixer
import copy


class VanillaQLearner:
    def __init__(self, mac: BasicMAC, scheme, logger, args) -> None:
        self.mac = mac
        self.mixer = QMixer(args)
        self.params = list(self.mac.parameters()) + list(self.mixer.parameters())
        self.optimizer = th.optim.RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )

        self.target_mac = copy.deepcopy(self.mac)
        self.target_mixer = copy.deepcopy(self.mixer)

    def train(batch: EpisodeBatch, t_env: int, episode_num: int) -> Any:
        # skipping the last item coz batch comes with +1 more step than the env episode_limit
        reward = batch['reward'][:, :-1]


    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimizer.state_dict(), "{}/optimizer.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path)))
