import os
import json
import copy
import fire
import datetime
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from transformers import get_cosine_schedule_with_warmup

from kformer.supervised.utils import Timer, profile
from kformer.supervised.utils import fleet_w2i, shipyard_w2i, fleet_dir


class KoreTensorDataset(Dataset):
    def __init__(self, data, data_path):
        super().__init__()
        self.df = data
        self.data_path = data_path
        self.row_indicies = self.df.index
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        row_index = self.row_indicies[i]
        row = self.df.loc[row_index]
        data = np.load(os.path.join(self.data_path, f"{row_index}.npz"), allow_pickle=True)
        data = data["arr_0"].item()
        return data


def CustomCollateFn(batch):
    batch_fmap = torch.vstack([x["fmap"].unsqueeze(0) for x in batch])
    batch_global_info = torch.vstack([x["global_info"].unsqueeze(0) for x in batch])
    batch_team_id = torch.vstack([x["team_id"].unsqueeze(0) for x in batch])
    batch_ship_pos_x = torch.tensor([x["ship_pos_x"] for x in batch]).long()
    batch_ship_pos_y = torch.tensor([x["ship_pos_y"] for x in batch]).long()
    max_team_fleet_len = max([len(x["team_fleet_pos"]) for x in batch])
    max_opp_fleet_len = max([len(x["opp_fleet_pos"]) for x in batch])
    
    def pad(a, value, l):
        a.extend([value] * l)
        return a
    
    batch_team_fleet_pos_x = [[pos_x for pos_x, pos_y in x["team_fleet_pos"]] for x in batch]
    batch_team_fleet_pos_x = [pad(a, 21, max_team_fleet_len-len(a)) for a in batch_team_fleet_pos_x]
    batch_team_fleet_pos_x = torch.tensor(batch_team_fleet_pos_x).long()
    batch_team_fleet_pos_y = [[pos_y for pos_x, pos_y in x["team_fleet_pos"]] for x in batch]
    batch_team_fleet_pos_y = [pad(a, 21, max_team_fleet_len-len(a)) for a in batch_team_fleet_pos_y]
    batch_team_fleet_pos_y = torch.tensor(batch_team_fleet_pos_y).long()
    batch_opp_fleet_pos_x = [[pos_x for pos_x, pos_y in x["opp_fleet_pos"]] for x in batch]
    batch_opp_fleet_pos_x = [pad(a, 21, max_opp_fleet_len-len(a)) for a in batch_opp_fleet_pos_x]
    batch_opp_fleet_pos_x = torch.tensor(batch_opp_fleet_pos_x).long()
    batch_opp_fleet_pos_y = [[pos_y for pos_x, pos_y in x["opp_fleet_pos"]] for x in batch]
    batch_opp_fleet_pos_y = [pad(a, 21, max_opp_fleet_len-len(a)) for a in batch_opp_fleet_pos_y]
    batch_opp_fleet_pos_y = torch.tensor(batch_opp_fleet_pos_y).long()
    
    def pad_embedding(a, value, l):
        if l == 0:
            return a
        return torch.cat([a, value.expand(l, value.shape[1])])
    
    batch_team_fleet_emb = [x["team_fleet_emb"] if x["team_fleet_emb"] is not None else torch.zeros((max_team_fleet_len, 288)) for x in batch]
    batch_team_fleet_emb = [pad_embedding(a, torch.zeros((1, a.shape[1])), max_team_fleet_len-a.shape[0]) for a in batch_team_fleet_emb]
    batch_team_fleet_emb = torch.vstack([a.unsqueeze(0) for a in batch_team_fleet_emb])
    batch_opp_fleet_emb = [x["opp_fleet_emb"] if x["opp_fleet_emb"] is not None else torch.zeros((max_opp_fleet_len, 288)) for x in batch]
    batch_opp_fleet_emb = [pad_embedding(a, torch.zeros((1, a.shape[1])), max_opp_fleet_len-a.shape[0]) for a in batch_opp_fleet_emb]
    batch_opp_fleet_emb = torch.vstack([a.unsqueeze(0) for a in batch_opp_fleet_emb])
    
    batch_team_fleet_mask = torch.tensor([[False] * max_team_fleet_len] * len(batch)).bool()
    for i in range(len(batch)):
        batch_team_fleet_mask[i, :len(batch[i]["team_fleet_pos"])] = True
    
    batch_opp_fleet_mask = torch.tensor([[False] * max_opp_fleet_len] * len(batch)).bool()
    for i in range(len(batch)):
        batch_opp_fleet_mask[i, :len(batch[i]["opp_fleet_pos"])] = True
    
    # max_action_len = max([len(x["action"]) for x in batch])
    max_action_len = 20
    batch_action_mask = torch.tensor([[0] * max_action_len] * len(batch)).long()
    for i in range(len(batch)):
        batch_action_mask[i, 0:len(batch[i]["action"])] = 1
        
    batch_action = [x["action"] for x in batch]
    batch_action = [pad(a, shipyard_w2i["PAD"], max_action_len-len(a)) for a in batch_action]
    batch_action = torch.tensor(batch_action).long() 
    
    return dict(
        fmap=batch_fmap,
        global_info=batch_global_info,
        team_id=batch_team_id,
        team_fleet_pos_x=batch_team_fleet_pos_x,
        team_fleet_pos_y=batch_team_fleet_pos_y,
        team_fleet_emb=batch_team_fleet_emb,
        team_fleet_mask=batch_team_fleet_mask,
        opp_fleet_pos_x=batch_opp_fleet_pos_x,
        opp_fleet_pos_y=batch_opp_fleet_pos_y,
        opp_fleet_emb=batch_opp_fleet_emb,
        opp_fleet_mask=batch_opp_fleet_mask,
        action=batch_action,
        action_mask=batch_action_mask,
        ship_pos_x=batch_ship_pos_x,
        ship_pos_y=batch_ship_pos_y,
    )


class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(output_dim) if bn else None

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        h = self.bn(h) if self.bn is not None else h
        return h


class SpatialEncoder(nn.Module):
    def __init__(self, input_ch, filters=32, layers=12):
        super().__init__()
        self.conv0 = TorusConv2d(input_ch, filters, (3, 3), True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), True) for _ in range(layers)])

    def forward(self, x):
        h = F.relu_(self.conv0(x))
        for block in self.blocks:
            h = F.relu_(h + block(h))
        return h


class KoreNet(nn.Module):
    def __init__(self, input_ch, batch_size):
        super().__init__()
        self.batch_size = batch_size

        self.scalar_encoder = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 288),
        )

        self.spatial_encoder = SpatialEncoder(input_ch, filters=32, layers=12)    
        self.up_projection = nn.Sequential(
            nn.Linear(32, 288),
            nn.LayerNorm(288)
        )

        self.pos_emb_x = nn.Embedding(22, 144, padding_idx=21)
        self.pos_emb_y = nn.Embedding(22, 144, padding_idx=21)
        self.domain_emb = nn.Embedding(5, 288, padding_idx=0) # SEP token is 0.
        self.seq_pos_emb = nn.Embedding(1000, 288)
        self.sep_token = nn.Parameter(torch.zeros(1, 288)).cuda()
        self.tgt_len = 20
        self.tgt_pos_emb = nn.Embedding(20, 288)
        self.team_emb = nn.Embedding(10, 288)

        self.transformer = nn.Transformer(
            d_model=288,
            nhead=4,
            num_encoder_layers=12,
            num_decoder_layers=4,
            dim_feedforward=1024,
        )
        self.fc = nn.Linear(288, 23)
    
    def prepare_input_seq(self, x):
        global_info = x["global_info"].cuda()
        scalar_token = self.scalar_encoder(global_info).unsqueeze(1) # shape (bs, 1, 288)
        
        fmap = x["fmap"].cuda()
        # print("count nans", torch.isnan(fmap).sum())
        # print("fmap", fmap)

        fmap = self.spatial_encoder(fmap)
        vis_tokens = fmap.view(fmap.shape[0], fmap.shape[1], -1).permute(0, 2, 1)
        # print("after conv", vis_tokens)

        vis_tokens = self.up_projection(vis_tokens) # shape (bs, 441, 288)
        # print("after up projection", vis_tokens)

        map_pos_x_embs = torch.tensor(np.arange(21)).unsqueeze(1).repeat(1, 21).view(1, -1).expand(fmap.shape[0], 21 * 21).long()
        map_pos_x_embs = self.pos_emb_x(map_pos_x_embs.cuda())
        map_pos_y_embs = torch.tensor(np.arange(21)).unsqueeze(0).repeat(fmap.shape[0], 21).long()
        map_pos_y_embs = self.pos_emb_y(map_pos_y_embs.cuda())
        map_pos_embs = torch.cat([map_pos_x_embs, map_pos_y_embs], dim=2)
        vis_tokens += map_pos_embs
        
        team_fleet_tokens = x["team_fleet_emb"].cuda() # shape (bs, n_fleet, 288)
        team_pos_x_embs = self.pos_emb_x(x["team_fleet_pos_x"].cuda())
        team_pos_y_embs = self.pos_emb_y(x["team_fleet_pos_y"].cuda())
        team_pos_embs = torch.cat([team_pos_x_embs, team_pos_y_embs], dim=2)
        team_fleet_tokens += team_pos_embs
        
        opp_fleet_tokens = x["opp_fleet_emb"].cuda() # shape (bs, n_fleet, 288)
        opp_pos_x_embs = self.pos_emb_x(x["opp_fleet_pos_x"].cuda())
        opp_pos_y_embs = self.pos_emb_y(x["opp_fleet_pos_y"].cuda())
        opp_pos_embs = torch.cat([opp_pos_x_embs, opp_pos_y_embs], dim=2)
        opp_fleet_tokens += opp_pos_embs
        
        # Full sequence
        sep_token = self.sep_token.unsqueeze(0).expand(fmap.shape[0], 1, -1) # shape (bs, 1, 288)

        # print("scalar", scalar_token)
        # print("vis", vis_tokens)
        # print("sep", sep_token)
        # print("team_fleet", team_fleet_tokens)
        # print("opp_fleet", opp_fleet_tokens)

        seq = torch.cat([scalar_token, vis_tokens, sep_token, team_fleet_tokens, sep_token, opp_fleet_tokens], dim=1)
        # seq = global_info + spatial_tokens + [SEP] + team_fleet_tokens + [SEP] + opp_fleet_tokens
        
        # domain coding:
        # global_info = 1
        # spatial = 2
        # fleet = 3
        # SEP token = 0
        domain_tokens = [1] + [2] * vis_tokens.shape[1] + [0] + [3] * team_fleet_tokens.shape[1] + [0] + [3] * opp_fleet_tokens.shape[1]
        domain_tokens = torch.tensor(domain_tokens).long().unsqueeze(0).expand(self.batch_size, -1)
        domain_embs = self.domain_emb(domain_tokens.cuda())
        seq += domain_embs
        
        seq_pos_embs = torch.arange(seq.shape[1]).unsqueeze(0).expand(seq.shape[0], -1)
        seq_pos_embs = self.seq_pos_emb(seq_pos_embs.cuda())
        seq += seq_pos_embs
        
        # Sequence mask
        team_fleet_mask = x["team_fleet_mask"].cuda()
        opp_fleet_mask = x["opp_fleet_mask"].cuda()
        global_info_mask = torch.tensor([True]).expand(seq.shape[0]).bool().unsqueeze(1).cuda()
        sep_mask = torch.tensor([False]).expand(seq.shape[0]).bool().unsqueeze(1).cuda()
        spatial_mask = torch.tensor(True).expand(seq.shape[0], 21 * 21).cuda()
        seq_mask = torch.cat([global_info_mask, spatial_mask, sep_mask, team_fleet_mask, sep_mask, opp_fleet_mask], dim=1)
        
        return seq, seq_mask
    
    def prepare_target_seq(self, x):
        pos_embs = torch.arange(self.tgt_len).long().cuda()
        pos_embs = self.tgt_pos_emb(pos_embs).unsqueeze(0).expand(self.batch_size, -1, -1)
        ship_pos_x = x["ship_pos_x"].cuda().unsqueeze(1)
        ship_pos_y = x["ship_pos_y"].cuda().unsqueeze(1)
        ship_pos_x_embs = self.pos_emb_x(ship_pos_x)
        ship_pos_y_embs = self.pos_emb_y(ship_pos_y)
        ship_pos_embs = torch.cat([ship_pos_x_embs, ship_pos_y_embs], dim=2).expand(-1, self.tgt_len, -1)
        team_embs = self.team_emb(x["team_id"].cuda()).expand(-1, self.tgt_len, -1)
        
        tgt = pos_embs + ship_pos_embs + team_embs
        tgt_mask = x["action_mask"].cuda().bool()
        return tgt, tgt_mask
        
    
    def forward(self, x):
        src, src_key_padding_mask = self.prepare_input_seq(x)
        tgt, tgt_key_padding_mask = self.prepare_target_seq(x)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        out = self.transformer(src, tgt, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        out = out.permute(1, 0, 2)
        out = self.fc(out)
        return out


class LightningModel(pl.LightningModule):
    def __init__(
        self, 
        lr, 
        weight_decay,
        warmup_steps,
        batch_size, 
        num_epochs, 
        num_gpus, 
        num_samples
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

        self.net = KoreNet(input_ch=11, batch_size=batch_size)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        action, action_mask = batch["action"], batch["action_mask"]
        out = self(batch)
        out = out.permute(0, 2, 1)
        loss = (self.ce_loss(out, action) * action_mask).sum() / action_mask.sum()
        self.log("train_loss", loss)
        return loss
    
    def training_step_end(self, training_step_outputs):
        (lr,) = self.scheduler.get_last_lr()
        self.log("lr", lr)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        total_steps = (
            self.num_samples
            * self.num_epochs
            // self.batch_size
            // self.num_gpus
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.num_epochs, eta_min=1e-8)
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
    lr=5e-4,
    weight_decay=1e-6,
    warmup_steps=4000,
    num_gpus=1,
    num_epochs=5,
    batch_size=32,
    train_csv_dir=os.path.join(base_path, "kore/train.csv"),
    tensor_dir=os.path.join(base_path, "kore/replays_tensors"),
):
    df = pd.read_csv(train_csv_dir)
    train_ds = KoreTensorDataset(data=df, data_path=tensor_dir)
    train_loader = DataLoader(train_ds, collate_fn=CustomCollateFn, batch_size=batch_size, num_workers=8, pin_memory=True, shuffle=True)

    # profiler = Timer()
    # for _ in range(25):
    #     batch = next(iter(train_loader))
    #     profile(profiler, "load a single batch from disk")

    model = LightningModel(
        lr=lr,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        batch_size=batch_size, 
        num_gpus=num_gpus,
        num_epochs=num_epochs,
        num_samples=len(train_ds),
    )
    
    log_dir = os.path.join(base_path, "logs/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(log_dir, exist_ok=False)
    print("Logs and model checkpoint will be saved to", log_dir)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        deterministic=True,
        precision=32,
        fast_dev_run=False,
        default_root_dir=log_dir,
        max_epochs=num_epochs, 
        track_grad_norm=2,
        enable_progress_bar=True
    )

    trainer.fit(model, train_loader)


if __name__ == "__main__":
    fire.Fire(main)

