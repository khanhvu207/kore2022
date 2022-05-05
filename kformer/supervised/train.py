import os
import json
import copy
import fire
import datetime
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

import datasets
from transformers import get_cosine_schedule_with_warmup

from kformer.supervised.utils import Timer, profile
from kformer.supervised.utils import fleet_w2i, shipyard_w2i, shipyard_i2w, fleet_dir


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
    
    # ATTENTION!
    # Pad = True, not-pad = False
    batch_team_fleet_mask = torch.tensor([[True] * max_team_fleet_len] * len(batch)).bool()
    for i in range(len(batch)):
        batch_team_fleet_mask[i, :len(batch[i]["team_fleet_pos"])] = False
    
    batch_opp_fleet_mask = torch.tensor([[True] * max_opp_fleet_len] * len(batch)).bool()
    for i in range(len(batch)):
        batch_opp_fleet_mask[i, :len(batch[i]["opp_fleet_pos"])] = False
    
    # max_action_len = max([len(x["action"]) for x in batch])
    max_action_len = 20
    batch_action_mask = torch.tensor([[True] * max_action_len] * len(batch)).bool()
    for i in range(len(batch)):
        batch_action_mask[i, :len(batch[i]["action"])] = False
        
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
    def __init__(self, input_ch, filters, layers):
        super().__init__()
        self.conv0 = TorusConv2d(input_ch, filters, (3, 3), bn=True)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), bn=True) for _ in range(layers)])
        # self._init_weight()

    def forward(self, x):
        h = self.conv0(x)
        for block in self.blocks:
            h = h + F.relu_(block(h))
        return h
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ScalarEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(3, 128),
            nn.BatchNorm1d(128), # Is bn neccessary?
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 288),
        )
    
    def forward(self, x):
        return self.fc(x)


class KoreNet(nn.Module):
    def __init__(self, input_ch):
        super().__init__()
        self.scalar_encoder = ScalarEncoder(3, 288)
        self.spatial_encoder = SpatialEncoder(input_ch, filters=32, layers=12)  
        self.up_projection = nn.Sequential(
            nn.Linear(32, 288),
            nn.LayerNorm(288)
        )

        # domain coding:
        # SEP token = 0
        # global info = 1
        # spatial = 2
        # fleet = 3
        self.domain_emb = nn.Embedding(4, 288, padding_idx=0)
        self.seq_pos_emb = nn.Embedding(1000, 288)
        self.sep_token = nn.Parameter(torch.zeros(1, 288)).cuda()
        self.tgt_len = 20
        self.tgt_pos_emb = nn.Embedding(20, 288)
        self.team_emb = nn.Embedding(10, 288) # Sanity check! Max team nr. is 10?
        self.pos_emb_x = nn.Embedding(22, 144, padding_idx=21)
        self.pos_emb_y = nn.Embedding(22, 144, padding_idx=21)

        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=288, nhead=8, dim_feedforward=2048, activation="relu"),
            num_layers=6,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=288, nhead=8, dim_feedforward=2048, activation="relu"), 
            num_layers=3
        )
        self.fc = nn.Linear(288, 23, bias=False)
    
    def prepare_input_seq(self, x):
        global_info = x["global_info"]
        self.batch_size = global_info.shape[0]
        scalar_token = self.scalar_encoder(global_info).unsqueeze(1) # shape (bs, 1, 288)
        # print("scalar", scalar_token)
        
        fmap = self.spatial_encoder(x["fmap"])
        spatial_tokens = fmap.view(fmap.shape[0], fmap.shape[1], -1).permute(0, 2, 1)
        spatial_tokens = self.up_projection(spatial_tokens) # shape (bs, 441, 288)
        # print("spatial", spatial_tokens)

        map_pos_x_embs = torch.arange(21, device="cuda").unsqueeze(1).repeat(1, 21).view(1, -1).expand(fmap.shape[0], 21 * 21)
        map_pos_x_embs = self.pos_emb_x(map_pos_x_embs.long()) # input to f.embedding is a Long tensor
        map_pos_y_embs = torch.arange(21, device="cuda").unsqueeze(0).repeat(fmap.shape[0], 21)
        map_pos_y_embs = self.pos_emb_y(map_pos_y_embs.long())
        map_pos_embs = torch.cat([map_pos_x_embs, map_pos_y_embs], dim=2)
        spatial_tokens += map_pos_embs
        
        team_fleet_tokens = x["team_fleet_emb"] # shape (bs, n_fleet, 288)
        team_pos_x_embs = self.pos_emb_x(x["team_fleet_pos_x"].long())
        team_pos_y_embs = self.pos_emb_y(x["team_fleet_pos_y"].long())
        team_pos_embs = torch.cat([team_pos_x_embs, team_pos_y_embs], dim=2)
        team_fleet_tokens += team_pos_embs
        
        opp_fleet_tokens = x["opp_fleet_emb"] # shape (bs, n_fleet, 288)
        opp_pos_x_embs = self.pos_emb_x(x["opp_fleet_pos_x"].long())
        opp_pos_y_embs = self.pos_emb_y(x["opp_fleet_pos_y"].long())
        opp_pos_embs = torch.cat([opp_pos_x_embs, opp_pos_y_embs], dim=2)
        opp_fleet_tokens += opp_pos_embs
        
        # Full sequence
        sep_token = self.sep_token.unsqueeze(0).expand(fmap.shape[0], 1, -1) # shape (bs, 1, 288)
        seq = torch.cat([scalar_token, spatial_tokens, sep_token, team_fleet_tokens, sep_token, opp_fleet_tokens], dim=1)
        
        domain_tokens = [1] + [2] * spatial_tokens.shape[1] + [0] + [3] * team_fleet_tokens.shape[1] + [0] + [3] * opp_fleet_tokens.shape[1]
        domain_tokens = torch.tensor(domain_tokens, device="cuda").unsqueeze(0).expand(self.batch_size, -1)
        domain_embs = self.domain_emb(domain_tokens.long())
        
        seq_pos_embs = torch.arange(seq.shape[1], device="cuda").unsqueeze(0).expand(seq.shape[0], -1)
        seq_pos_embs = self.seq_pos_emb(seq_pos_embs.long())
        seq = seq + domain_embs + seq_pos_embs
        
        # Sequence mask
        # ATTENTION!
        # Pad = True, not-pad = False
        global_info_mask = torch.tensor([False], device="cuda", dtype=torch.bool).expand(seq.shape[0]).unsqueeze(1)
        sep_mask = torch.tensor([True], device="cuda", dtype=torch.bool).expand(seq.shape[0]).unsqueeze(1)
        spatial_mask = torch.tensor(False, device="cuda", dtype=torch.bool).expand(seq.shape[0], 21 * 21)
        team_fleet_mask = x["team_fleet_mask"].bool()
        opp_fleet_mask = x["opp_fleet_mask"].bool()
        seq_mask = torch.cat([global_info_mask, spatial_mask, sep_mask, team_fleet_mask, sep_mask, opp_fleet_mask], dim=1)

        return seq, seq_mask
    
    def prepare_target_seq(self, x):
        pos_embs = torch.arange(self.tgt_len, device="cuda")
        pos_embs = self.tgt_pos_emb(pos_embs.long()).unsqueeze(0).expand(self.batch_size, -1, -1)

        ship_pos_x = x["ship_pos_x"].unsqueeze(1)
        ship_pos_y = x["ship_pos_y"].unsqueeze(1)
        ship_pos_x_embs = self.pos_emb_x(ship_pos_x.long())
        ship_pos_y_embs = self.pos_emb_y(ship_pos_y.long())
        ship_pos_embs = torch.cat([ship_pos_x_embs, ship_pos_y_embs], dim=2).expand(-1, self.tgt_len, -1)
        
        team_embs = self.team_emb(x["team_id"]).expand(-1, self.tgt_len, -1)

        tgt = pos_embs + ship_pos_embs + team_embs

        tgt_mask = x["action_mask"]
        return tgt, tgt_mask
    
    def get_encoder_output(self, src, src_key_padding_mask):
        return self.transformer_encoder(
            src, 
            src_key_padding_mask=src_key_padding_mask
        )
    
    def get_decoder_output(self, memory, tgt, tgt_key_padding_mask, memory_key_padding_mask):
        return self.transformer_decoder(
            tgt=tgt, 
            memory=memory, 
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

    def forward(self, x):
        src, src_key_padding_mask = self.prepare_input_seq(x)
        tgt, tgt_key_padding_mask = self.prepare_target_seq(x)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        memory = self.get_encoder_output(src, src_key_padding_mask)
        out = self.get_decoder_output(
            memory, 
            tgt, 
            tgt_key_padding_mask, 
            memory_key_padding_mask=src_key_padding_mask
        )

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

        self.net = KoreNet(input_ch=11)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
    
    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        action, action_mask = batch["action"], batch["action_mask"]
        out = self(batch)
        out = out.permute(0, 2, 1)
        loss_mask = (1 - action_mask.long()) # Flip the mask
        loss = (self.ce_loss(out, action) * loss_mask).sum() / loss_mask.sum()
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        action, action_mask = batch["action"], batch["action_mask"]
        out = self(batch)
        out = out.permute(0, 2, 1) 
        loss_mask = (1 - action_mask.long())
        loss = (self.ce_loss(out, action) * loss_mask).sum() / loss_mask.sum()
        self.log("val_loss", loss)
        return {
            "out": out.detach().cpu(),
            "action": action.detach().cpu()
        }
    
    def validation_epoch_end(self, validation_step_outputs):
        outs = {
            k: [dic[k] for dic in validation_step_outputs]
            for k in validation_step_outputs[0]
        }

        preds = torch.cat(outs["out"], dim=0) # shape (n, 23, 20)
        preds = torch.softmax(preds.permute(0, 2, 1), dim=1).argmax(dim=1)
        actions = torch.cat(outs["action"], dim=0) # shape (n, 20)
        preds = preds.tolist()
        actions = actions.tolist()

        def preprocess(pred, action):
            try:
                first_padding_pos = action.index(shipyard_w2i["PAD"])
            except:
                first_padding_pos = None
            
            if first_padding_pos is not None:
                pred = pred[:first_padding_pos]
                action = action[:first_padding_pos]
            
            pred = list(map(shipyard_i2w.get, pred))
            action = list(map(shipyard_i2w.get, action))
            return pred, action
        
        predictions = []
        references = []
        for pred, action in zip(preds, actions):
            new_pred, new_action = preprocess(pred, action)
            predictions.append(new_pred)
            references.append([new_action])

        bleu = datasets.load_metric("bleu")
        results = bleu.compute(predictions=predictions, references=references)
        self.log("bleu", results["bleu"])
        self.log("1-gram precision", results["precisions"][0])

        flatten_predictions = np.concatenate(predictions)
        flatten_references = np.concatenate([x[0] for x in references])
        assert flatten_predictions.shape == flatten_references.shape
        accuracy = np.mean(flatten_predictions == flatten_references)
        self.log("accuracy", accuracy)

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
    debug=True,
    train_csv_dir=os.path.join(base_path, "kore/train.csv"),
    tensor_dir=os.path.join(base_path, "kore/replays_tensors"),
):
    df = pd.read_csv(train_csv_dir)
    train_df, val_df, _, _ = train_test_split(
        df, df["TokenizedAction"], stratify=df["NumFleet"], test_size=0.1, random_state=2022
    )
    train_df.sort_values(by=["NumFleet"], inplace=True)
    val_df.sort_values(by=["NumFleet"], inplace=True)
    num_teams = len(set(df["TeamNameId"]))
    print("Train length:", len(train_df))
    print("Validation length:", len(val_df))
    print("Nr. teams:", num_teams)

    train_ds = KoreTensorDataset(data=train_df, data_path=tensor_dir)
    train_loader = DataLoader(
        train_ds, 
        collate_fn=CustomCollateFn, 
        batch_size=batch_size, 
        num_workers=8, 
        pin_memory=True, 
        shuffle=True
    )
    val_ds = KoreTensorDataset(data=val_df, data_path=tensor_dir)
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
    )
    
    log_dir = os.path.join(base_path, "logs/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    if not debug:
        os.makedirs(log_dir, exist_ok=False)
        print("Logs and model checkpoint will be saved to", log_dir)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=num_gpus,
        deterministic=True,
        precision=32,
        strategy="ddp",
        detect_anomaly=debug,
        log_every_n_steps=1 if debug else 50,
        fast_dev_run=5 if debug else False,
        default_root_dir=log_dir,
        max_epochs=num_epochs, 
        track_grad_norm=2,
        enable_progress_bar=True,
        logger=not debug,
        checkpoint_callback=not debug
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    fire.Fire(main)

