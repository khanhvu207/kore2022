import os
import gc
import sys
import json
import pickle
import zipfile
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
from pympler.asizeof import asizeof

from kformer.supervised.utils import Timer, profile
from kformer.supervised.utils import fleet_w2i, shipyard_w2i, fleet_dir


class KoreDataset(Dataset):
    def __init__(self, data, device, plan_encoder, ep, debug=False):
        super().__init__()
        self.df = data
        self.device = device
        self.ep = ep
        self.debug = debug
        self.plan_encoder = plan_encoder
        self.plan_encoder.eval()
    
    def __len__(self):
        return len(self.df)
    
    def get_obs_at_step(self, ep_id, step):
        obs = self.ep[ep_id]["steps"][step][0]["observation"]
        return obs
    
    def get_feature_map(self, obs, player):
        def normalize(x: torch.tensor) -> torch.tensor:
            return (x - x.min()) / x.max()
        
        kore = torch.tensor(obs["kore"]).float().view(1, 21, 21)
    
        team_shipyard = torch.zeros((1, 21, 21))
        team_shipyard_num_ship = torch.zeros((1, 21, 21))
        team_shipyard_max_spawn = torch.zeros((1, 21, 21))
        for shipyard_id, info in obs["players"][player][1].items():
            pos, num_ship, turn_controlled = info
            pos_x = pos // 21
            pos_y = pos % 21
            team_shipyard[0, pos_x, pos_y] = 1
            team_shipyard_num_ship[0, pos_x, pos_y] = num_ship
            team_shipyard_max_spawn[0, pos_x, pos_y] = int(np.log2(max(turn_controlled, 2)-1))+1
            
        team_fleet = torch.zeros((1, 21, 21))
        team_fleet_kore = torch.zeros((1, 21, 21))
        for fleet_id, info in obs["players"][player][2].items():
            pos, fleet_kore, fleet_size, direction, plan = info
            pos_x = pos // 21
            pos_y = pos % 21
            team_fleet[0, pos_x, pos_y] = fleet_size
            team_fleet_kore[0, pos_x, pos_y] = fleet_kore
            
        opponent = 1 - player
        opp_shipyard = torch.zeros((1, 21, 21))
        opp_shipyard_num_ship = torch.zeros((1, 21, 21))
        opp_shipyard_max_spawn = torch.zeros((1, 21, 21))
        for shipyard_id, info in obs["players"][opponent][1].items():
            pos, num_ship, turn_controlled = info
            pos_x = pos // 21
            pos_y = pos % 21
            opp_shipyard[0, pos_x, pos_y] = 1
            opp_shipyard_num_ship[0, pos_x, pos_y] = num_ship
            opp_shipyard_max_spawn[0, pos_x, pos_y] = int(np.log2(max(turn_controlled, 2)-1))+1
        
        opp_fleet = torch.zeros((1, 21, 21))
        opp_fleet_kore = torch.zeros((1, 21, 21))
        for fleet_id, info in obs["players"][opponent][2].items():
            pos, fleet_kore, fleet_size, direction, plan = info
            pos_x = pos // 21
            pos_y = pos % 21
            opp_fleet[0, pos_x, pos_y] = fleet_size
            opp_fleet_kore[0, pos_x, pos_y] = fleet_kore
            
        shipyard = torch.vstack([team_shipyard, opp_shipyard])
        shipyard_num_ship = torch.vstack([team_shipyard_num_ship, opp_shipyard_num_ship])
        shipyard_max_spawn = torch.vstack([team_shipyard_max_spawn, opp_shipyard_max_spawn])
        fleet = torch.vstack([team_fleet, opp_fleet])
        fleet_kore = torch.vstack([team_fleet_kore, opp_fleet_kore])
        
        kore = normalize(kore)
        shipyard_num_ship = normalize(shipyard_num_ship)
        shipyard_max_spawn = (shipyard_max_spawn / 10).float()
        fleet = normalize(fleet)
        fleet_kore = normalize(fleet_kore)
        return torch.vstack([kore, shipyard, shipyard_num_ship, shipyard_max_spawn, fleet, fleet_kore])
    
    def get_fleet_embedding(self, obs, player):
        profiler = Timer()
        
        team_fleet_pos = []
        team_fleet_plan = []
        
        for fleet_id, info in obs["players"][player][2].items():
            pos, fleet_kore, fleet_size, direction, plan = info
            pos_x = pos // 21
            pos_y = pos % 21
            team_fleet_pos.append((pos_x, pos_y))
            if len(plan) == 0:
                plan = fleet_dir[direction]
            plan = " ".join(["CLS", " ".join(list(plan)), "EOS"])
            plan = list(map(fleet_w2i.get, plan.split(" ")))
            team_fleet_plan.append(plan)
        
        profile(profiler, "loop1", self.debug)
        
        opponent = 1 - player
        opp_fleet_pos = []
        opp_fleet_plan = []
        for fleet_id, info in obs["players"][opponent][2].items():
            pos, fleet_kore, fleet_size, direction, plan = info
            pos_x = pos // 21
            pos_y = pos % 21
            opp_fleet_pos.append((pos_x, pos_y))
            if len(plan) == 0:
                plan = fleet_dir[direction]
            plan = " ".join(["CLS", " ".join(list(plan)), "EOS"])
            plan = list(map(fleet_w2i.get, plan.split(" ")))
            opp_fleet_plan.append(plan)
            
        profile(profiler, "loop2", self.debug)
        
        def pad(t, l):
            if len(t) < l:
                t.extend([fleet_w2i["PAD"]] * (l - len(t)))
            return t
        
        team_fleet_embs = None
        opp_fleet_embs = None
        
        with torch.inference_mode():
            if len(team_fleet_plan) > 0:
                team_max_plan_len = max([len(t) for t in team_fleet_plan])
                team_fleet_plan = torch.tensor([pad(t, team_max_plan_len) for t in team_fleet_plan]).long()
                _, team_fleet_embs = self.plan_encoder(team_fleet_plan.to(self.device))
                team_fleet_embs = team_fleet_embs.cpu()
            profile(profiler, "inference1", self.debug)

            if len(opp_fleet_plan) > 0:
                opp_max_plan_len = max([len(t) for t in opp_fleet_plan])
                opp_fleet_plan = torch.tensor([pad(t, opp_max_plan_len) for t in opp_fleet_plan]).long()
                _, opp_fleet_embs = self.plan_encoder(opp_fleet_plan.to(self.device))
                opp_fleet_embs = opp_fleet_embs.cpu()
            profile(profiler, "inference2", self.debug)
        
        return team_fleet_pos, team_fleet_embs, opp_fleet_pos, opp_fleet_embs
    
    def get_shipyard_pos(self, obs, target_id, player, row):
        pos_x = None
        pos_y = None
        for shipyard_id, info in obs["players"][player][1].items():
            if shipyard_id == target_id:
                pos, num_ship, turn_controlled = info
                pos_x = pos // 21
                pos_y = pos % 21
                break
        
        assert pos_x is not None, f"{obs['players'][player][1]}, {target_id}, {row}"
        assert pos_y is not None
        return pos_x, pos_y
    
    def __getitem__(self, i):
        profiler = Timer()
        row = self.df.loc[i]
        player = row["Player"]
        step = row["Step"]
        ep_id = str(row["EpisodeId"])
        profile(profiler, "initialize")
        
        obs = self.get_obs_at_step(ep_id, step) # Get observation from step-1
        ship_pos_x, ship_pos_y = self.get_shipyard_pos(
            obs, 
            row["ShipyardId"], 
            player,
            row,
        )

        fmap = self.get_feature_map(obs, player)
        profile(profiler, "get_feature_map", self.debug)
        
        action = list(map(shipyard_w2i.get, row["TokenizedAction"].split(" ")))
        team_fleet_pos, team_fleet_emb, opp_fleet_pos, opp_fleet_emb = self.get_fleet_embedding(obs, player)
        profile(profiler, "get_fleet_embedding", self.debug)
        
        player_kore = obs["players"][player][0] / 100000
        opp_kore = obs["players"][1-player][0] / 100000
        step = step / 400
        team_id = torch.tensor(row["TeamNameId"]).long()
        global_info = torch.tensor([step, player_kore, opp_kore]).float()
        profile(profiler, "finalizing", self.debug)
        
        return dict(
            ep_id=row["EpisodeId"], 
            team_id=team_id,
            global_info=global_info,
            fmap=fmap,
            team_fleet_pos=team_fleet_pos,
            team_fleet_emb=team_fleet_emb,
            opp_fleet_pos=opp_fleet_pos,
            opp_fleet_emb=opp_fleet_emb,
            action=action,
            ship_pos_x=ship_pos_x,
            ship_pos_y=ship_pos_y,
        )


base_path = "/var/scratch/kvu400"

def main(
    json_dir=os.path.join(base_path, "kore/replays"), 
    encoder_dir=os.path.join(base_path, "kore/encoder_jit.pt"), 
    train_csv_dir=os.path.join(base_path, "kore/train.csv"),
    target_dir=os.path.join(base_path, "kore/dataset.zip")
):
    df = pd.read_csv(train_csv_dir)
    print(df.head())

    ep = defaultdict(dict)
    for file in os.listdir(json_dir):
        with open(os.path.join(json_dir, file)) as f:
            data = json.load(f)
        ep_id = str(file.split('.')[0])
        ep[ep_id] = data
    
    plan_encoder = torch.jit.load(encoder_dir)
    print(plan_encoder.code)

    train_ds = KoreDataset(data=df, device="cuda:0", plan_encoder=plan_encoder, ep=ep, debug=False)

    z = zipfile.PyZipFile(target_dir, mode='w')
    for i in tqdm(range(len(train_ds))):
        out = train_ds[i]
        f = f"{i}.npz"
        np.savez_compressed(f, out)
        z.write(f)
        os.remove(f)

    print(len(z.namelist()))
    z.close()

    del df, ep, plan_encoder
    gc.collect()


if __name__ == "__main__":
    main()