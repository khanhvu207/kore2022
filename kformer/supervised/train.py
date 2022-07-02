import os
import json
import copy
import fire
import pickle
import datetime
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader

import datasets
from transformers import get_cosine_schedule_with_warmup

from kformer.supervised.model import KoreNet
from kformer.supervised.utils import Timer, profile
from kformer.supervised.utils import fleet_w2i, shipyard_w2i, shipyard_i2w, fleet_dir, action_encoder


class KoreDataset(Dataset):
    def __init__(self, df, ep, debug=False):
        super().__init__()
        self.df = df
        self.indicies = self.df.index
        self.ep = ep
        self.debug = debug
        self.team_le = LabelEncoder().fit(self.df["TeamName"])
    
    def __len__(self):
        return len(self.df)
    
    def get_obs_at_step(self, ep_id, step):
        obs = self.ep[ep_id]["steps"][step][0]["observation"]
        return obs
    
    def _normalize(self, x, low=-1.0, high=1.0, eps=1e-5) -> torch.tensor:
        return (high - low) * (x - x.min()) / (x.max() + eps) + low
    
    def _standardize(self, x, eps=1e-5) -> torch.tensor:
        return (x - x.mean()) / (x.std() + eps)
    
    def get_feature_map(self, obs, player):
        kore = torch.tensor(obs["kore"], dtype=torch.float).view(1, 21, 21)
        
        team_shipyard = torch.zeros((1, 21, 21))
        team_shipyard_num_ship = torch.zeros((1, 21, 21))
        team_shipyard_max_spawn = torch.zeros((1, 21, 21))
        team_fleet = torch.zeros((1, 21, 21))
        team_fleet_cargo = torch.zeros((1, 21, 21))
        
        for shipyard_id, info in obs["players"][player][1].items():
            pos, num_ship, turn_controlled = info
            pos_x = pos // 21
            pos_y = pos % 21
            team_shipyard[0, pos_x, pos_y] = 1
            team_shipyard_num_ship[0, pos_x, pos_y] = num_ship
            team_shipyard_max_spawn[0, pos_x, pos_y] = int(np.log2(max(turn_controlled, 2)-1))+1
            
        for fleet_id, info in obs["players"][player][2].items():
            pos, fleet_kore, fleet_size, direction, plan = info
            pos_x = pos // 21
            pos_y = pos % 21
            team_fleet[0, pos_x, pos_y] = fleet_size
            team_fleet_cargo[0, pos_x, pos_y] = fleet_kore
            
        opponent = 1 - player
        opp_shipyard = torch.zeros((1, 21, 21))
        opp_shipyard_num_ship = torch.zeros((1, 21, 21))
        opp_shipyard_max_spawn = torch.zeros((1, 21, 21))
        opp_fleet = torch.zeros((1, 21, 21))
        opp_fleet_cargo = torch.zeros((1, 21, 21))
        
        for shipyard_id, info in obs["players"][opponent][1].items():
            pos, num_ship, turn_controlled = info
            pos_x = pos // 21
            pos_y = pos % 21
            opp_shipyard[0, pos_x, pos_y] = 1
            opp_shipyard_num_ship[0, pos_x, pos_y] = num_ship
            opp_shipyard_max_spawn[0, pos_x, pos_y] = int(np.log2(max(turn_controlled, 2)-1))+1
        
        for fleet_id, info in obs["players"][opponent][2].items():
            pos, fleet_kore, fleet_size, direction, plan = info
            pos_x = pos // 21
            pos_y = pos % 21
            opp_fleet[0, pos_x, pos_y] = fleet_size
            opp_fleet_cargo[0, pos_x, pos_y] = fleet_kore
            
        shipyard = torch.vstack([team_shipyard, opp_shipyard])
        shipyard_num_ship = torch.vstack([team_shipyard_num_ship, opp_shipyard_num_ship])
        shipyard_max_spawn = torch.vstack([team_shipyard_max_spawn, opp_shipyard_max_spawn])
        fleet = torch.vstack([team_fleet, opp_fleet])
        fleet_cargo = torch.vstack([team_fleet_cargo, opp_fleet_cargo])
        
        cur_max_kore = kore.max()
        kore = self._normalize(kore / 500).float()
        shipyard = self._normalize(shipyard).float()
        shipyard_num_ship = self._normalize(shipyard_num_ship / 1000).float()
        shipyard_max_spawn = self._normalize(shipyard_max_spawn / 10).float()
        fleet = self._normalize(fleet / 1000).float()
        fleet_cargo = self._normalize(fleet_cargo / cur_max_kore).float()
        return torch.vstack([kore, shipyard, shipyard_num_ship, shipyard_max_spawn, fleet, fleet_cargo])
    
    def _tokenize_fleet_plan(self, plan, max_len):
        tokens = list(plan) + ["EOS"] + ["PAD"] * (max_len - len(plan) + 1)
        mask = [False] * (1 + len(plan)) + [True] * (max_len - len(plan) + 1)
        return list(map(fleet_w2i.get, tokens)), mask
    
    def get_fleet_info(self, obs, player):
        team_fleet_pos = [(21, 21)] # Prepend a ghost fleet to avoid some exceptions...
        team_fleet_plan = ["N"]
        team_plan_max_len = 0
        for fleet_id, info in obs["players"][player][2].items():
            pos, fleet_kore, fleet_size, direction, plan = info
            plan = fleet_dir[direction] if len(plan) == 0 else plan
            pos_x = pos // 21
            pos_y = pos % 21
            team_fleet_pos.append((pos_x, pos_y))
            team_fleet_plan.append(plan)
            team_plan_max_len = max(team_plan_max_len, len(plan))
        
        opponent = 1 - player
        opp_fleet_pos = [(21, 21)]
        opp_fleet_plan = ["N"]
        opp_plan_max_len = 0
        for fleet_id, info in obs["players"][opponent][2].items():
            pos, fleet_kore, fleet_size, direction, plan = info
            plan = fleet_dir[direction] if len(plan) == 0 else plan
            pos_x = pos // 21
            pos_y = pos % 21
            opp_fleet_pos.append((pos_x, pos_y))
            opp_fleet_plan.append(plan)
            opp_plan_max_len = max(opp_plan_max_len, len(plan))
        
        team_fleet = list(map(lambda f: self._tokenize_fleet_plan(f, team_plan_max_len), team_fleet_plan))
        opp_fleet = list(map(lambda f: self._tokenize_fleet_plan(f, opp_plan_max_len), opp_fleet_plan))
        return team_fleet, team_fleet_pos, opp_fleet, opp_fleet_pos
    
    def __getitem__(self, i):
        profiler = Timer()
        
        idx = self.indicies[i]
        row = self.df.loc[idx]
        player = row["Player"]
        step = row["Step"]
        ep_id = str(row["EpisodeId"])
        profile(profiler, "initialize", self.debug)
        
        obs = self.get_obs_at_step(ep_id, step)
        fmap = self.get_feature_map(obs, player)
        profile(profiler, "get_feature_map", self.debug)
        
        team_fleet, team_fleet_pos, opp_fleet, opp_fleet_pos = self.get_fleet_info(obs, player)
        team_fleet_plan, team_fleet_plan_mask = map(list, zip(*team_fleet))
        opp_fleet_plan, opp_fleet_plan_mask = map(list, zip(*opp_fleet))
        team_fleet_plan = torch.tensor(team_fleet_plan, dtype=torch.long)
        team_fleet_plan_mask = torch.tensor(team_fleet_plan_mask, dtype=torch.bool)
        opp_fleet_plan = torch.tensor(opp_fleet_plan, dtype=torch.long)
        opp_fleet_plan_mask = torch.tensor(opp_fleet_plan_mask, dtype=torch.bool)
        profile(profiler, "get_fleet_info", self.debug)
        
        player_kore = obs["players"][player][0] / 100000
        opp_kore = obs["players"][1-player][0] / 100000
        team_id = self.team_le.transform([row["TeamName"]]).item()

        shipyard_pos_x = row["ShipyardPosX"]
        shipyard_pos_y = row["ShipyardPosY"]
        shipyard_nr_ship = int(row["ShipyardNrShip"]) / 500
        shipyard_turn_controlled = row["ShipyardTurnControlled"]
        
        action = action_encoder[row["Action"]]
        spawn_nr = int(row["Spawn_nr"]) # Multi-class prediction target
        launch_nr = int(row["Launch_nr"]) / 149 # Regression target
        is_spawn = row["Action"] == "SPAWN"
        is_launch = row["Action"] == "LAUNCH"
        
        plan = ["CLS"] + list(str(row["Plan"]) if str(row["Plan"]) != "nan" else "") + ["EOS"]
        plan = torch.tensor(list(map(shipyard_w2i.get, plan)), dtype=torch.long)
        profile(profiler, "finalizing", self.debug)
        
        return dict(
            step=step,
            team_id=team_id,
            fmap=fmap,
            team_kore=player_kore,
            team_fleet_pos=team_fleet_pos,
            team_fleet_plan=team_fleet_plan,
            team_fleet_plan_mask=team_fleet_plan_mask,
            opp_kore=opp_kore,
            opp_fleet_pos=opp_fleet_pos,
            opp_fleet_plan=opp_fleet_plan,
            opp_fleet_plan_mask=opp_fleet_plan_mask,
            shipyard_pos_x=shipyard_pos_x,
            shipyard_pos_y=shipyard_pos_y,
            shipyard_nr_ship=shipyard_nr_ship,
            shipyard_turn_controlled=shipyard_turn_controlled,
            action=action,
            spawn_nr=spawn_nr,
            launch_nr=launch_nr,
            is_spawn=is_spawn,
            is_launch=is_launch,
            plan=plan,
        )


def CustomCollateFn(batch):
    batch_size = len(batch)
    
    batch_step = torch.tensor([x["step"] for x in batch], dtype=torch.long).unsqueeze(1) # shape (bs, 1)
    batch_team_id = torch.tensor([x["team_id"] for x in batch], dtype=torch.long).unsqueeze(1)  # shape (bs, 1)
    batch_fmap = torch.vstack([x["fmap"].unsqueeze(0) for x in batch]) # shape (bs, ch, 21, 21)
    
    team_fleet_plans = [x["team_fleet_plan"] for x in batch]
    team_fleet_masks = [x["team_fleet_plan_mask"] for x in batch]
    team_max_fleet_size = max([x.shape[0] for x in team_fleet_plans])
    team_max_fleet_plan = max([x.shape[1] for x in team_fleet_plans])
    batch_team_fleet_mask = torch.ones((batch_size, team_max_fleet_size), dtype=torch.bool)
    batch_team_fleet_plan = torch.full((batch_size, team_max_fleet_size, team_max_fleet_plan), fleet_w2i["PAD"], dtype=torch.long)
    batch_team_fleet_plan_mask = torch.ones((batch_size, team_max_fleet_size, team_max_fleet_plan), dtype=torch.bool)
    for batch_idx, (fleet_plans, fleet_masks) in enumerate(zip(team_fleet_plans, team_fleet_masks)):
        nr_fleet = fleet_plans.shape[0]
        plan_size = fleet_plans.shape[1]
        batch_team_fleet_mask[batch_idx, 1:nr_fleet] = False # Ignore the dummy fleet
        batch_team_fleet_plan[batch_idx, :nr_fleet, :plan_size].copy_(fleet_plans)
        batch_team_fleet_plan_mask[batch_idx, :nr_fleet, :plan_size].copy_(fleet_masks)
    
    batch_team_fleet_plan[:, :, 0] = fleet_w2i["CLS"]
    batch_team_fleet_plan_mask[:, :, 0] = False
    batch_team_kore = torch.tensor([x["team_kore"] for x in batch], dtype=torch.float).unsqueeze(1)
    
    team_fleet_pos = [x["team_fleet_pos"] for x in batch]
    team_fleet_pos_x = [torch.tensor([x for x, y in pos]) for pos in team_fleet_pos]
    team_fleet_pos_y = [torch.tensor([y for x, y in pos]) for pos in team_fleet_pos]
    batch_team_fleet_pos_x = torch.full((batch_size, team_max_fleet_size), 21, dtype=torch.long)
    batch_team_fleet_pos_y = torch.full((batch_size, team_max_fleet_size), 21, dtype=torch.long)
    for batch_idx, (pos_x, pos_y) in enumerate(zip(team_fleet_pos_x, team_fleet_pos_y)):
        batch_team_fleet_pos_x[batch_idx, :pos_x.shape[0]].copy_(pos_x)
        batch_team_fleet_pos_y[batch_idx, :pos_y.shape[0]].copy_(pos_y)
    
    opp_fleet_plans = [x["opp_fleet_plan"] for x in batch]
    opp_fleet_masks = [x["opp_fleet_plan_mask"] for x in batch]
    opp_max_fleet_size = max([x.shape[0] for x in opp_fleet_plans])
    opp_max_fleet_plan = max([x.shape[1] for x in opp_fleet_plans])
    batch_opp_fleet_mask = torch.ones((batch_size, opp_max_fleet_size), dtype=torch.bool)
    batch_opp_fleet_plan = torch.full((batch_size, opp_max_fleet_size, opp_max_fleet_plan), fleet_w2i["PAD"], dtype=torch.long)
    batch_opp_fleet_plan_mask = torch.ones((batch_size, opp_max_fleet_size, opp_max_fleet_plan), dtype=torch.bool)
    for batch_idx, (fleet_plans, fleet_masks) in enumerate(zip(opp_fleet_plans, opp_fleet_masks)):
        nr_fleet = fleet_plans.shape[0]
        plan_size = fleet_plans.shape[1]
        batch_opp_fleet_mask[batch_idx, 1:nr_fleet] = False
        batch_opp_fleet_plan[batch_idx, :nr_fleet, :plan_size].copy_(fleet_plans)
        batch_opp_fleet_plan_mask[batch_idx, :nr_fleet, :plan_size].copy_(fleet_masks)
    
    batch_opp_fleet_plan[:, :, 0] = fleet_w2i["CLS"]
    batch_opp_fleet_plan_mask[:, :, 0] = False
    batch_opp_kore = torch.tensor([x["opp_kore"] for x in batch], dtype=torch.float).unsqueeze(1)
    
    opp_fleet_pos = [x["opp_fleet_pos"] for x in batch]
    opp_fleet_pos_x = [torch.tensor([x for x, y in pos]) for pos in opp_fleet_pos]
    opp_fleet_pos_y = [torch.tensor([y for x, y in pos]) for pos in opp_fleet_pos]
    batch_opp_fleet_pos_x = torch.full((batch_size, opp_max_fleet_size), 21, dtype=torch.long)
    batch_opp_fleet_pos_y = torch.full((batch_size, opp_max_fleet_size), 21, dtype=torch.long)
    for batch_idx, (pos_x, pos_y) in enumerate(zip(opp_fleet_pos_x, opp_fleet_pos_y)):
        batch_opp_fleet_pos_x[batch_idx, :pos_x.shape[0]].copy_(pos_x)
        batch_opp_fleet_pos_y[batch_idx, :pos_y.shape[0]].copy_(pos_y)
    
    batch_shipyard_pos_x = torch.tensor([x["shipyard_pos_x"] for x in batch], dtype=torch.long).unsqueeze(1)
    batch_shipyard_pos_y = torch.tensor([x["shipyard_pos_y"] for x in batch], dtype=torch.long).unsqueeze(1)
    batch_shipyard_nr_ship = torch.tensor([x["shipyard_nr_ship"] for x in batch], dtype=torch.float).unsqueeze(1)
    batch_shipyard_turn_controlled = torch.tensor([x["shipyard_turn_controlled"] for x in batch], dtype=torch.long).unsqueeze(1)
    batch_action = torch.tensor([x["action"] for x in batch], dtype=torch.long)
    
    batch_spawn_nr = torch.tensor([x["spawn_nr"] for x in batch], dtype=torch.long)
    batch_launch_nr = torch.tensor([x["launch_nr"] for x in batch], dtype=torch.float).unsqueeze(1)
    batch_is_spawn = torch.tensor([x["is_spawn"] for x in batch], dtype=torch.long).unsqueeze(1)
    batch_is_launch = torch.tensor([x["is_launch"] for x in batch], dtype=torch.long).unsqueeze(1)

    plans = [x["plan"] for x in batch]
    max_plan_len = 12 # max([plan.shape[0] for plan in plans])
    batch_plan = torch.full((batch_size, max_plan_len), shipyard_w2i["PAD"], dtype=torch.long)
    batch_plan_mask = torch.zeros((batch_size, max_plan_len), dtype=torch.long)
    for batch_idx, plan in enumerate(plans):
        plan_len = plan.shape[0]
        batch_plan[batch_idx, :plan_len].copy_(plan)
        batch_plan_mask[batch_idx, :plan_len] = 1

    # print(batch_plan)
    # print(batch_plan_mask)
    
    # Sanity check
    # assert batch_step.shape == (batch_size, 1)
    # assert batch_team_id.shape == (batch_size, 1)
    # assert batch_fmap.shape == (batch_size, 11, 21, 21)
    # assert batch_team_fleet_mask.shape == (batch_size, team_max_fleet_size)
    # assert batch_team_fleet_plan.shape == (batch_size, team_max_fleet_size, team_max_fleet_plan)
    # assert batch_team_fleet_plan_mask.shape == (batch_size, team_max_fleet_size, team_max_fleet_plan)
    # assert batch_opp_fleet_mask.shape == (batch_size, opp_max_fleet_size)
    # assert batch_opp_fleet_plan.shape == (batch_size, opp_max_fleet_size, opp_max_fleet_plan)
    # assert batch_opp_fleet_plan_mask.shape == (batch_size, opp_max_fleet_size, opp_max_fleet_plan)
    # assert batch_shipyard_pos_x.shape == (batch_size, 1)
    # assert batch_shipyard_pos_y.shape == (batch_size, 1)
    # assert batch_shipyard_nr_ship.shape == (batch_size, 1)
    # assert batch_shipyard_turn_controlled.shape == (batch_size, 1)
    # assert batch_cardinality.shape == (batch_size, 1)
    # assert batch_plan.shape == (batch_size, max_plan_len)
    # assert batch_plan_mask.shape == (batch_size, max_plan_len)
    
    return dict(
        step=batch_step,
        team_id=batch_team_id,
        fmap=batch_fmap,
        team_kore=batch_team_kore,
        team_fleet_mask=batch_team_fleet_mask,
        team_fleet_plan=batch_team_fleet_plan,
        team_fleet_plan_mask=batch_team_fleet_plan_mask,
        team_fleet_pos_x=batch_team_fleet_pos_x,
        team_fleet_pos_y=batch_team_fleet_pos_y,
        opp_kore=batch_opp_kore,
        opp_fleet_mask=batch_opp_fleet_mask,
        opp_fleet_plan=batch_opp_fleet_plan,
        opp_fleet_plan_mask=batch_opp_fleet_plan_mask,
        opp_fleet_pos_x=batch_opp_fleet_pos_x,
        opp_fleet_pos_y=batch_opp_fleet_pos_y,
        shipyard_pos_x=batch_shipyard_pos_x,
        shipyard_pos_y=batch_shipyard_pos_y,
        shipyard_nr_ship=batch_shipyard_nr_ship,
        shipyard_turn_controlled=batch_shipyard_turn_controlled,
        action=batch_action,
        spawn_nr=batch_spawn_nr,
        launch_nr=batch_launch_nr,
        is_spawn=batch_is_spawn,
        is_launch=batch_is_launch,
        plan=batch_plan,
        plan_mask=batch_plan_mask,
    )


class LightningModel(pl.LightningModule):
    def __init__(
        self, 
        lr, 
        weight_decay,
        warmup_steps,
        batch_size, 
        num_epochs, 
        num_gpus, 
        num_samples,
        token_dim,
        debug=False
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_gpus = num_gpus
        self.num_samples = num_samples
        self.debug = debug

        self.net = KoreNet(token_dim, debug)

        self.eps = 1e-7
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        out = self(batch)
        num_outputs = len(out["action_logits"])
        
        action_loss = 0
        spawn_nr_loss = 0
        launch_nr_loss = 0
        plan_loss = 0
        ground_truth = batch["plan"][:, 1:] # Shifted one position to the left
        ground_truth_mask = batch["plan_mask"][:, 1:]

        for i in range(num_outputs):
            layer_action_loss = nn.CrossEntropyLoss()(out["action_logits"][i], batch["action"])
            layer_spawn_nr_loss = nn.CrossEntropyLoss(reduction="none")(out["spawn_nr_logits"][i], batch["spawn_nr"]).unsqueeze(1)
            layer_spawn_nr_loss = (layer_spawn_nr_loss * batch["is_spawn"]).sum() / (batch["is_spawn"].sum() + self.eps)
            layer_launch_nr_loss = nn.MSELoss(reduction="none")(out["launch_nr_logits"][i], batch["launch_nr"])
            layer_launch_nr_loss = (layer_launch_nr_loss * batch["is_launch"]).sum() / (batch["is_launch"].sum() + self.eps)
            layer_pred_plan = out["plan_logits"][i].permute(0, 2, 1)
            layer_plan_loss = nn.CrossEntropyLoss(reduction="none")(layer_pred_plan, ground_truth)
            layer_plan_loss = (layer_plan_loss * ground_truth_mask).sum() / ground_truth_mask.sum()

            action_loss += layer_action_loss
            spawn_nr_loss += layer_spawn_nr_loss
            launch_nr_loss += layer_launch_nr_loss
            plan_loss += layer_plan_loss

        # action_loss = nn.CrossEntropyLoss()(out["action_logit"], batch["action"])
        
        # spawn_nr_loss = nn.CrossEntropyLoss(reduction="none")(out["spawn_nr_logit"], batch["spawn_nr"]).unsqueeze(1)
        # spawn_nr_loss = (spawn_nr_loss * batch["is_spawn"]).sum() / (batch["is_spawn"].sum() + self.eps)

        # launch_nr_loss = nn.MSELoss(reduction="none")(out["launch_nr_logit"], batch["launch_nr"])
        # launch_nr_loss = (launch_nr_loss * batch["is_launch"]).sum() / (batch["is_launch"].sum() + self.eps)
        
        # pred_plan = out["plan_logit"].permute(0, 2, 1)
        # ground_truth = batch["plan"][:, 1:] # Shifted one position to the left
        # ground_truth_mask = batch["plan_mask"][:, 1:]
        # plan_loss = nn.CrossEntropyLoss(reduction="none")(pred_plan, ground_truth)
        # plan_loss = (plan_loss * ground_truth_mask).sum() / ground_truth_mask.sum()

        loss = action_loss + spawn_nr_loss + launch_nr_loss + plan_loss

        self.log("train/loss", loss)
        self.log("train/action_loss", action_loss)
        self.log("train/spawn_nr_loss", spawn_nr_loss)
        self.log("train/launch_nr_loss", launch_nr_loss)
        self.log("train/plan_loss", plan_loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        out = self(batch)
        num_outputs = len(out["action_logits"])
        
        action_loss = 0
        spawn_nr_loss = 0
        launch_nr_loss = 0
        plan_loss = 0
        ground_truth = batch["plan"][:, 1:] # Shifted one position to the left
        ground_truth_mask = batch["plan_mask"][:, 1:]

        for i in range(num_outputs):
            layer_action_loss = nn.CrossEntropyLoss()(out["action_logits"][i], batch["action"])
            layer_spawn_nr_loss = nn.CrossEntropyLoss(reduction="none")(out["spawn_nr_logits"][i], batch["spawn_nr"]).unsqueeze(1)
            layer_spawn_nr_loss = (layer_spawn_nr_loss * batch["is_spawn"]).sum() / (batch["is_spawn"].sum() + self.eps)
            layer_launch_nr_loss = nn.MSELoss(reduction="none")(out["launch_nr_logits"][i], batch["launch_nr"])
            layer_launch_nr_loss = (layer_launch_nr_loss * batch["is_launch"]).sum() / (batch["is_launch"].sum() + self.eps)
            layer_pred_plan = out["plan_logits"][i].permute(0, 2, 1)
            layer_plan_loss = nn.CrossEntropyLoss(reduction="none")(layer_pred_plan, ground_truth)
            layer_plan_loss = (layer_plan_loss * ground_truth_mask).sum() / ground_truth_mask.sum()

            action_loss += layer_action_loss
            spawn_nr_loss += layer_spawn_nr_loss
            launch_nr_loss += layer_launch_nr_loss
            plan_loss += layer_plan_loss

        # action_loss = nn.CrossEntropyLoss()(out["action_logit"], batch["action"])
        
        # spawn_nr_loss = nn.CrossEntropyLoss(reduction="none")(out["spawn_nr_logit"], batch["spawn_nr"]).unsqueeze(1)
        # spawn_nr_loss = (spawn_nr_loss * batch["is_spawn"]).sum() / (batch["is_spawn"].sum() + self.eps)

        # launch_nr_loss = nn.MSELoss(reduction="none")(out["launch_nr_logit"], batch["launch_nr"])
        # launch_nr_loss = (launch_nr_loss * batch["is_launch"]).sum() / (batch["is_launch"].sum() + self.eps)
        
        # pred_plan = out["plan_logit"].permute(0, 2, 1)
        # ground_truth = batch["plan"][:, 1:] # Shifted one position to the left
        # ground_truth_mask = batch["plan_mask"][:, 1:]
        # plan_loss = nn.CrossEntropyLoss(reduction="none")(pred_plan, ground_truth)
        # plan_loss = (plan_loss * ground_truth_mask).sum() / ground_truth_mask.sum()

        loss = action_loss + spawn_nr_loss + launch_nr_loss + plan_loss

        self.log("val/loss", loss)
        self.log("val/action_loss", action_loss)
        self.log("val/spawn_nr_loss", spawn_nr_loss)
        self.log("val/launch_nr_loss", launch_nr_loss)
        self.log("val/plan_loss", plan_loss)
        # return dict(
        #     action_logit=out["action_logit"].detach().cpu(),
        #     action=batch["action"].detach().cpu(),
        #     spawn_nr_logit=out["spawn_nr_logit"].detach().cpu(),
        #     spawn_nr=batch["spawn_nr"].detach().cpu(),
        #     plan_logit=out["plan_logit"].detach().cpu(),
        #     plan=ground_truth.detach().cpu(),
        #     plan_mask=ground_truth_mask.detach().cpu(),
        #     is_launch=batch["is_launch"].detach().cpu(),
        #     is_spawn=batch["is_spawn"].detach().cpu(),
        # )
        return dict(
            action_logit=out["action_logits"][-1].detach().cpu(),
            action=batch["action"].detach().cpu(),
            spawn_nr_logit=out["spawn_nr_logits"][-1].detach().cpu(),
            spawn_nr=batch["spawn_nr"].detach().cpu(),
            plan_logit=out["plan_logits"][-1].detach().cpu(),
            plan=ground_truth.detach().cpu(),
            plan_mask=ground_truth_mask.detach().cpu(),
            is_launch=batch["is_launch"].detach().cpu(),
            is_spawn=batch["is_spawn"].detach().cpu(),
        )
    
    def validation_epoch_end(self, validation_step_outputs):
        outs = {
            k: [dic[k] for dic in validation_step_outputs]
            for k in validation_step_outputs[0]
        }

        action_logit = torch.cat(outs["action_logit"], dim=0)
        pred_action = action_logit.softmax(dim=1).argmax(dim=1)
        action = torch.cat(outs["action"], dim=0)
        action_acc = (pred_action == action).float().mean().item()
        self.log("val/action_accuracy", action_acc)

        pred_action = pred_action.numpy().tolist()
        action = action.numpy().tolist()
        action_label = ["IDLE", "SPAWN", "LAUNCH"]
        for x in range(3):
            tp = 0
            pred_count = 0
            true_count = 0
            
            for i in range(len(pred_action)):
                pred_count += pred_action[i] == x
                true_count += action[i] == x
                if pred_action[i] == x:
                    tp += pred_action[i] == action[i]
            
            precision = tp / (pred_count + self.eps)
            recall = tp / (true_count + self.eps)
            
            if self.debug:
                print(action_label[x], f"precision={precision}, recall={recall}")

            self.log(f"val/{action_label[x]}_precision", precision)
            self.log(f"val/{action_label[x]}_recall", recall) 

        spawn_nr_logit = torch.cat(outs["spawn_nr_logit"], dim=0)
        pred_spawn_nr = spawn_nr_logit.softmax(dim=1).argmax(dim=1)
        spawn_nr = torch.cat(outs["spawn_nr"], dim=0)
        is_spawn = torch.cat(outs["is_spawn"], dim=0).squeeze(1)
        spawn_nr_acc = (((pred_spawn_nr == spawn_nr) * is_spawn).sum() / is_spawn.sum()).item()
        self.log("val/spawn_nr_accuracy", spawn_nr_acc)

        plan_logit = torch.cat(outs["plan_logit"], dim=0)
        pred_plan = plan_logit.softmax(dim=2).argmax(dim=2)
        plan = torch.cat(outs["plan"], dim=0)
        plan_mask = torch.cat(outs["plan_mask"], dim=0)
        is_launch = torch.cat(outs["is_launch"], dim=0).squeeze(1)
        plan_acc_per_sample = ((pred_plan == plan).float() * plan_mask).sum(dim=1) / plan_mask.sum(dim=1)
        plan_acc = ((plan_acc_per_sample * is_launch).sum() / is_launch.sum()).item()
        self.log("val/plan_token_accuracy", plan_acc)

        if self.debug:
            print("Action accuracy", action_acc)
            print("Spawn_nr accuracy", spawn_nr_acc)
            print("Plan token accuracy", plan_acc)

    def training_step_end(self, training_step_outputs):
        (lr,) = self.scheduler.get_last_lr()
        self.log("lr", lr)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = (
            self.num_samples
            * self.num_epochs
            // self.batch_size
            // self.num_gpus
        )
        print(f"Number of warm-up steps: {self.warmup_steps}/{total_steps}")
        self.scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=total_steps
        )
        lr_scheduler_config = {
            "scheduler": self.scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}


base_path = "/var/scratch/kvu400"
def main(
    lr=3e-4,
    weight_decay=1e-2,
    warmup_steps=1000,
    gradient_clip_val=0.5,
    num_gpus=1,
    num_epochs=20,
    batch_size=32,
    token_dim=128,
    debug=True,
    train_csv_dir=os.path.join(base_path, "kore/train.csv"),
    replay_dir=os.path.join(base_path, "kore/replays.pkl"),
):
    df = pd.read_csv(train_csv_dir)
    train_df, val_df, _, _ = train_test_split(
        df, df["Action"], stratify=df["Action"], test_size=0.1, random_state=2022
    )
    num_teams = len(set(df["TeamName"]))
    print("Train length:", len(train_df))
    print("Validation length:", len(val_df))
    print("Nr. teams:", num_teams)

    with open(replay_dir, "rb") as f:
        ep = pickle.load(f)

    print("Nr. of episodes", len(ep))

    train_ds = KoreDataset(df=train_df, ep=ep)
    train_loader = DataLoader(
        train_ds, 
        collate_fn=CustomCollateFn, 
        batch_size=batch_size, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=True
    )
    val_ds = KoreDataset(df=val_df, ep=ep)
    val_loader = DataLoader(
        val_ds, 
        collate_fn=CustomCollateFn, 
        batch_size=batch_size, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=False
    )

    model = LightningModel(
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        batch_size=batch_size, 
        num_gpus=num_gpus,
        num_epochs=num_epochs,
        num_samples=len(train_ds),
        token_dim=token_dim,
        debug=debug
    )
    
    log_dir = os.path.join(base_path, "logs/" + datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    log_dir += f"-lr-{lr}"
    log_dir += f"-gpu-{num_gpus}"
    log_dir += f"-bs-{batch_size}"

    if not debug:
        os.makedirs(log_dir, exist_ok=False)
        print("Logs and model checkpoint will be saved to", log_dir)

    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        dirpath=log_dir,
    )
    callback_list = [checkpoint_callback] if not debug else []

    if not debug:
        wandb_logger = WandbLogger(
            project="kore2022",
            entity="kaggle-kvu",
            name=f"lr-{lr}-weight_decay-{weight_decay}-nepochs-{num_epochs}",
            save_dir=log_dir,
            offline=debug,
            sync_tensorboard=True,
        )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        deterministic=True,
        precision=32, # Use fp32 for now
        strategy="ddp" if num_gpus>1 else None,
        detect_anomaly=debug,
        log_every_n_steps=1 if debug else 50,
        fast_dev_run=5 if debug else False,
        default_root_dir=log_dir,
        gradient_clip_val=gradient_clip_val, 
        max_epochs=num_epochs, 
        track_grad_norm=2,
        num_sanity_val_steps=False, # not working for some reasons
        enable_progress_bar=False,
        logger=False if debug else wandb_logger,
        enable_checkpointing=not debug,
        callbacks=callback_list,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    fire.Fire(main)

