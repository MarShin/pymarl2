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
        self.optimiser = th.optim.RMSprop(
            params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps
        )

        self.target_mac = copy.deepcopy(self.mac)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.last_target_update_episode = 0

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int) -> Any:
        # Get the relevant quantities
        # skipping the last item coz batch comes with +1 more step than the env episode_limit
        reward = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        avail_actions = batch["avail_actions"]

        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions
        ).unsqueeze(3)
        # Remove the last dim

        # Calculate the Q-Values necessary for the target

        # We don't need the first timesteps Q-Value estimate for calculating targets
        # Concat across time

        # Mask out unavilable actions

        # Max over target Q-Values
        #     Get actions that maximise live Q [later]

        # Mix

        # Calculate 1-step Q-Learning targets

        # Td-error

        # 0-out teh targets that came from padded data

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (
            episode_num - self.last_target_update_episode
        ) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num


def _update_targets(self):
    self.target_mac.load_state(self.mac)
    self.target_mixer.load_state_dict(self.mixer.state_dict())
    self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.cuda()
        self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/optimiser.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.mixer.load_state_dict(th.load("{}/mixer.th".format(path)))
