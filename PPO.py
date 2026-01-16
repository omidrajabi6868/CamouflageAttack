import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPO:
    def __init__(self):
        self.action_values = torch.tensor([0.001, 0.01, 0.8, 1.0, 1.25, 10, 100])
        self.num_actions = len(self.action_values)
        self.controller = PPOController(state_dim=4, NUM_ACTIONS=self.num_actions)
        self.buffer =  RolloutBuffer()

    def initial_state(self, proposals, gt_instances, patch):
        state, _, _ = self.compute_rewards(proposals, gt_instances, patch)
        return state

    def sample_actions(self, logits_dict):
        actions = {}
        log_probs = {}

        for k in ["lr", "rpn_cls", "rpn_loc", "feature", "seg", "roi_cls", "roi_loc", "mask"]:
            dist = Categorical(logits=logits_dict[k])
            a = dist.sample()
            actions[k] = a
            log_probs[k] = dist.log_prob(a)

        return actions, log_probs

    def compute_rewards(self, proposals, gt_instances, patch):
        # Extract metrics
        N_boxes = sum(proposals[0].objectness_logits > 0)/(len(gt_instances[0].gt_classes) + 1e-8)
        max_conf = torch.max(proposals[0].objectness_logits)
        sum_conf = torch.sum(proposals[0].objectness_logits[proposals[0].objectness_logits > 0])

        # Reward (IMPORTANT)
        reward = (
            -N_boxes
            -0.005 * sum_conf
            -1.0 * max_conf
        )

        next_state = torch.tensor([
            N_boxes,
            max_conf,
            sum_conf,
            0 if patch.grad == None else patch.grad.norm().item()
        ])

        done = float(N_boxes == 0)

        return next_state, reward, done

    def compute_gae(self, rewards, values, dones, gamma=0.95, lam=0.95):
        advantages = []
        gae = 0
        values = values + [0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def ppo_update(self, model, optimizer, clip_eps=0.2, vf_coef=0.5, ent_coef=0.01):
        model.train()
        states = torch.stack(self.buffer.states)
        rewards = self.buffer.rewards
        dones = self.buffer.dones

        with torch.no_grad():
            values = self.buffer.values
            advantages = self.compute_gae(rewards, values, dones)
            advantages = torch.tensor(advantages)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            returns = advantages + torch.tensor(values)

        logits = model(states)
        new_values = logits["value"]

        policy_loss = 0
        entropy = 0

        for k in ["lr", "rpn_cls", "rpn_loc", "feature", "seg", "roi_cls", "roi_loc", "mask"]:
            dist = Categorical(logits=logits[k])
    
            # Convert actions to tensor
            actions_tensor = torch.tensor(self.buffer.actions[k], dtype=torch.long, device=logits[k].device)
            old_log_probs_tensor = torch.tensor(self.buffer.log_probs[k], dtype=torch.float32, device=logits[k].device)
            
            # Compute new log_prob
            new_log_prob = dist.log_prob(actions_tensor)

            # Ratio for PPO
            ratio = torch.exp(new_log_prob - old_log_probs_tensor)
            clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

            policy_loss += -torch.mean(torch.min(ratio * advantages, clipped * advantages))

            entropy += dist.entropy().mean()

        value_loss = nn.MSELoss()(new_values, returns)

        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



class PPOController(nn.Module):
    def __init__(self, state_dim, NUM_ACTIONS):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.lr_head = nn.Linear(128, NUM_ACTIONS)
        self.rpn_cls_head = nn.Linear(128, NUM_ACTIONS)
        self.rpn_loc_head = nn.Linear(128, NUM_ACTIONS)
        self.feature_head = nn.Linear(128, NUM_ACTIONS)
        self.seg_head = nn.Linear(128, NUM_ACTIONS)
        self.roi_cls_head = nn.Linear(128, NUM_ACTIONS)
        self.roi_loc_head = nn.Linear(128, NUM_ACTIONS)
        self.mask_head = nn.Linear(128, NUM_ACTIONS)

        self.value_head = nn.Linear(128, 1)

    def forward(self, state):
        x = self.shared(state)
        return {
            "lr": self.lr_head(x),
            "rpn_cls": self.rpn_cls_head(x),
            "rpn_loc": self.rpn_loc_head(x),
            "feature": self.feature_head(x),
            "seg": self.seg_head(x),
            "roi_cls": self.roi_cls_head(x),
            "roi_loc": self.roi_loc_head(x),
            "mask": self.mask_head(x),
            "value": self.value_head(x).squeeze(-1)
        }

class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = {"lr": [], "rpn_cls": [], "rpn_loc": [], "feature": [], "seg": [], "roi_cls": [], "roi_loc": [], "mask": []}
        self.log_probs = {"lr": [], "rpn_cls": [], "rpn_loc": [], "feature": [], "seg": [], "roi_cls": [], "roi_loc": [], "mask": []}
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()
