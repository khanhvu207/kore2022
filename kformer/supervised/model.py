import torch
import torch.nn as nn
import torch.nn.functional as F

from kformer.supervised.utils import Timer, profile


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
        self.up_projection = nn.Conv2d(filters, 128, (1, 1), bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        h = self.conv0(x)
        for block in self.blocks:
            h = h + F.relu_(block(h))
        return self.up_projection(h)


class ScalarEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, x):
        return self.fc(x)
    
    
class FlightPlanEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(18, 128)
        self.pos_embedding = nn.Embedding(20, 128)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128, 
                nhead=1,
                dim_feedforward=256,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=2,
        )
    
    def forward(self, fleet_plan, fleet_plan_mask):
        bs, nr_fleet, plan_len = fleet_plan.shape
        fleet_plan = fleet_plan.reshape(bs * nr_fleet, -1)
        fleet_plan_mask = fleet_plan_mask.view(bs * nr_fleet, -1)
        seq_emb = self.token_embedding(fleet_plan)
        seq_emb = seq_emb.permute(1, 0, 2)
        out = self.encoder(seq_emb, src_key_padding_mask=fleet_plan_mask).permute(1, 0, 2)
        return out.reshape(bs, nr_fleet, plan_len, 128)[:, :, 0, :]
    

class FusionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=128, 
                nhead=4,
                dim_feedforward=256,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=4,
        )
        
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=128, 
                nhead=4,
                dim_feedforward=256,
                activation="gelu",
                norm_first=True,
            ),
            num_layers=2,
        )
        self.sep_token = nn.Parameter(nn.init.normal_(torch.empty(1, 128)))
        self.tgt_token = nn.Parameter(nn.init.normal_(torch.empty(20, 128)))
        self.team_fleet = nn.Parameter(nn.init.normal_(torch.empty(1, 128)))
        self.opp_fleet = nn.Parameter(nn.init.normal_(torch.empty(1, 128)))

        self.pos_x = nn.Embedding(22, 64, padding_idx=21)
        self.pos_y = nn.Embedding(22, 64, padding_idx=21)
        self.team_id = nn.Embedding(10, 128)
        self.step = nn.Embedding(400, 128)
        self.tgt_pos = nn.Embedding(20, 128)
    
    def forward(self, fmap_emb, team_fleet_emb, opp_fleet_emb, x):
        team_pos_emb = torch.cat([self.pos_x(x["team_fleet_pos_x"]), self.pos_x(x["team_fleet_pos_y"])], dim=2)
        team_fleet_emb = team_fleet_emb + self.team_fleet.expand(team_fleet_emb.shape) + team_pos_emb
        opp_pos_emb = torch.cat([self.pos_x(x["opp_fleet_pos_x"]), self.pos_x(x["opp_fleet_pos_y"])], dim=2)
        opp_fleet_emb = opp_fleet_emb + self.opp_fleet.expand(opp_fleet_emb.shape) + opp_pos_emb

        seq = torch.cat(
            [
                fmap_emb, # TODO: Are there embeddings need to add to this?
                self.sep_token.expand((fmap_emb.shape[0], 1, 128)),
                team_fleet_emb, 
                self.sep_token.expand((fmap_emb.shape[0], 1, 128)),
                opp_fleet_emb,
            ], 
            dim=1
        )
        
        team_fleet_mask = x["team_fleet_mask"]
        opp_fleet_mask = x["opp_fleet_mask"]
        bs, seq_len, _ = seq.shape
        _, fmap_len, _ = fmap_emb.shape
        _, team_fleet_len, _ = team_fleet_emb.shape
        _, opp_fleet_len, _ = opp_fleet_emb.shape
        seq_mask = torch.full((bs, seq_len), False, dtype=torch.bool, device=seq.device)
        seq_mask[:, fmap_len+1:fmap_len+1+team_fleet_len].copy_(team_fleet_mask)
        seq_mask[:, fmap_len+1+team_fleet_len+1:].copy_(opp_fleet_mask)
        
        seq = seq.permute(1, 0, 2)
        memory = self.encoder(seq, src_key_padding_mask=seq_mask)

        tgt_seq = self.tgt_token.expand((bs, 20, 128))
        tgt_seq = tgt_seq + self.team_id(x["team_id"]).expand((-1, 20, -1)) + self.step(x["step"]).expand((-1, 20, -1))
        tgt_seq = tgt_seq.permute(1, 0, 2)
        
        out = self.decoder(tgt=tgt_seq, memory=memory, memory_key_padding_mask=seq_mask)
        return out.permute(1, 0, 2)
    
    
class KoreNet(nn.Module):
    def __init__(self, debug):
        super().__init__()
        self.debug = debug
        if self.debug:
            torch.set_printoptions(profile="full")

        self.spatial_encoder = SpatialEncoder(input_ch=11, filters=32, layers=12)
        self.flight_plan_encoder = FlightPlanEncoder()
        self.fusion_transformer = FusionTransformer()

        self.action_head = nn.Linear(128, 3, bias=False)
        self.cardinality_head = nn.Linear(128, 1, bias=False)
        self.plan_head = nn.Linear(128, 18, bias=False)
    
    def forward(self, x):
        profiler = Timer()
        profile(profiler, "init", self.debug)
        
        team_fleet_emb = self.flight_plan_encoder(x["team_fleet_plan"], x["team_fleet_plan_mask"]) # shape (bs, nr_fleet, 128)
        opp_fleet_emb = self.flight_plan_encoder(x["opp_fleet_plan"], x["opp_fleet_plan_mask"]) # shape (bs, nr_fleet, 128)
        profile(profiler, "flight_plan_encoder", self.debug)
        
        fmap_emb = self.spatial_encoder(x["fmap"])
        bs, ch, w, h = fmap_emb.shape
        fmap_emb = fmap_emb.view(bs, ch, -1).permute(0, 2, 1) # shape (bs, 441, 128)
        profile(profiler, "spatial_encoder", self.debug)

        out = self.fusion_transformer(fmap_emb, team_fleet_emb, opp_fleet_emb, x)
        profile(profiler, "fusion_transformer", self.debug)
        
        if self.debug:
            assert fmap_emb.isnan().any().item() == False, "fmap_emb contains nan!"
            assert team_fleet_emb.isnan().any().item() == False, "team_fleet_emb contains nan!"
            assert opp_fleet_emb.isnan().any().item() == False, "opp_fleet_emb contains nan!"
            assert out.isnan().any().item() == False, "out contains nan!"

        return dict(
            action_logit=self.action_head(out[:, 0, :]),
            cardinality_logit=self.cardinality_head(out[:, 0, :]),
            plan_logit=self.plan_head(out[:, 1:, :])
        )