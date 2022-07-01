import torch
import torch.nn as nn
import torch.nn.functional as F

from kformer.supervised.utils import Timer, profile
from kformer.supervised.utils import fleet_w2i


class TorusConv2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, use_gn):
        super().__init__()
        self.edge_size = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size, bias=False)
        self.gn = nn.GroupNorm(16, 32)
        self.use_gn = use_gn

    def forward(self, x):
        h = torch.cat([x[:,:,:,-self.edge_size[1]:], x, x[:,:,:,:self.edge_size[1]]], dim=3)
        h = torch.cat([h[:,:,-self.edge_size[0]:], h, h[:,:,:self.edge_size[0]]], dim=2)
        h = self.conv(h)
        if self.use_gn:
            h = self.gn(h)
        return h


class SpatialEncoder(nn.Module):
    def __init__(self, token_dim, input_ch, filters, layers):
        super().__init__()
        self.token_dim = token_dim
        self.conv0 = TorusConv2d(input_ch, filters, (3, 3), use_gn=False)
        self.blocks = nn.ModuleList([TorusConv2d(filters, filters, (3, 3), use_gn=True) for _ in range(layers)])
        self.up_projection = nn.Conv2d(filters, self.token_dim, (1, 1), bias=True)

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
        
        h = self.up_projection(h)
        return h


class ScalarEncoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

        for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)    

    def forward(self, x):
        return self.fc(x)


class FusionTransformer(nn.Module):
    def __init__(self, token_dim):
        super().__init__()
        self.token_dim = token_dim

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=self.token_dim, 
                nhead=4,
                dim_feedforward=512,
                activation="gelu",
                norm_first=True,
                dropout=0.1,
            ),
            num_layers=8,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=self.token_dim, 
                nhead=4,
                dim_feedforward=512,
                activation="gelu",
                norm_first=True,
                dropout=0.1,
            ),
            num_layers=4,
        )
        self.src_ln = nn.LayerNorm(self.token_dim)
        self.tgt_ln = nn.LayerNorm(self.token_dim)

        self.max_tgt_plan_len = 11
        self.sep_token = nn.Parameter(nn.init.normal_(torch.empty(1, self.token_dim)))
        self.tgt_cls_token = nn.Parameter(nn.init.normal_(torch.empty(1, self.token_dim)))
        
        self.global_info_marker = nn.Parameter(nn.init.normal_(torch.empty(1, self.token_dim))) 
        self.fmap_marker = nn.Parameter(nn.init.normal_(torch.empty(1, self.token_dim))) 
        self.fleet_marker = nn.Parameter(nn.init.normal_(torch.empty(1, self.token_dim))) 
        self.team_marker = nn.Parameter(nn.init.normal_(torch.empty(1, self.token_dim)))
        self.opp_marker = nn.Parameter(nn.init.normal_(torch.empty(1, self.token_dim)))

        self.pos_x = nn.Embedding(22, self.token_dim // 2, padding_idx=21)
        self.pos_y = nn.Embedding(22, self.token_dim // 2, padding_idx=21)
        self.team_id = nn.Embedding(10, self.token_dim)
        self.step = nn.Embedding(400, self.token_dim)

        self.plan_token_embedding = nn.Embedding(18, self.token_dim, padding_idx=fleet_w2i["PAD"])
        self.plan_pos_embedding = nn.Embedding(20, self.token_dim)
        self.plan_fc = nn.Linear(self.token_dim, self.token_dim)

    def _get_autoregressive_attn_mask(self, tgt_len, device):
        # Create an upper triangular matrix filled with -inf
        mask = torch.triu(torch.ones(tgt_len, tgt_len, device=device), diagonal=1)
        mask[mask.bool()] = -float("inf")
        return mask

    def _get_fleet_plan_emb(self, fleet_plan, fleet_plan_mask):
        bs, nr_fleet, plan_len = fleet_plan.shape
        
        seq_emb = self.plan_token_embedding(fleet_plan) # shape (bs, nr_fleet, plan_len, self.token_dim)
        pos_emb = torch.arange(plan_len, dtype=torch.long, device=seq_emb.device)
        pos_emb = self.plan_pos_embedding(pos_emb) # shape (plan_len, self.token_dim)
        pos_emb = pos_emb.expand(seq_emb.shape)
        seq_emb = seq_emb + pos_emb
        
        seq_emb = self.plan_fc(seq_emb)
        
        inverted_mask = 1 - fleet_plan_mask.long().unsqueeze(3).expand(seq_emb.shape)
        seq_emb = seq_emb * inverted_mask
        fleet_emb = seq_emb.sum(dim=2) # Bag of words

        return fleet_emb # shape (bs, nr_fleet, self.token_dim)

    def forward(
        self, 
        fmap_emb, 
        team_info_emb, 
        team_fleet_plan,
        team_fleet_plan_mask, 
        opp_info_emb, 
        opp_fleet_plan,
        opp_fleet_plan_mask,
        x
    ):
        team_fleet_emb = self._get_fleet_plan_emb(team_fleet_plan, team_fleet_plan_mask)
        opp_fleet_emb = self._get_fleet_plan_emb(opp_fleet_plan, opp_fleet_plan_mask)

        # Replace some tokens with zero vector
        if self.training is True:
            randomized_indicies = torch.randperm(441)
            masked_indices = randomized_indicies[:69] # I like 69 :D
            fmap_emb[:, masked_indices, :] = fmap_emb[:, masked_indices, :].fill_(0) 
        
        # Add positional embeddings
        self.cur_device = fmap_emb.device
        map_pos_x_embs = torch.arange(21, device=self.cur_device).unsqueeze(1).repeat(1, 21).view(1, -1).expand(fmap_emb.shape[0], 21 * 21)
        map_pos_x_embs = self.pos_x(map_pos_x_embs.long())
        map_pos_y_embs = torch.arange(21, device=self.cur_device).unsqueeze(0).repeat(fmap_emb.shape[0], 21)
        map_pos_y_embs = self.pos_y(map_pos_y_embs.long())
        map_pos_embs = torch.cat([map_pos_x_embs, map_pos_y_embs], dim=2)
        fmap_emb = fmap_emb + map_pos_embs

        team_pos_emb = torch.cat([self.pos_x(x["team_fleet_pos_x"]), self.pos_x(x["team_fleet_pos_y"])], dim=2)
        team_fleet_emb = team_fleet_emb + self.team_marker.expand(team_fleet_emb.shape) + team_pos_emb
        opp_pos_emb = torch.cat([self.pos_x(x["opp_fleet_pos_x"]), self.pos_x(x["opp_fleet_pos_y"])], dim=2)
        opp_fleet_emb = opp_fleet_emb + self.opp_marker.expand(opp_fleet_emb.shape) + opp_pos_emb

        team_info_emb = team_info_emb + self.team_marker.expand(team_info_emb.shape)
        opp_info_emb = opp_info_emb + self.opp_marker.expand(opp_info_emb.shape)

        # Concat everything
        seq = torch.cat(
            [
                fmap_emb + self.fmap_marker.expand(fmap_emb.shape),
                team_info_emb + self.global_info_marker.expand(team_info_emb.shape),
                opp_info_emb + self.global_info_marker.expand(opp_info_emb.shape),
                team_fleet_emb + self.fleet_marker.expand(team_fleet_emb.shape), 
                opp_fleet_emb  + self.fleet_marker.expand(opp_fleet_emb.shape),
            ], 
            dim=1
        )
        
        # Create input mask
        team_fleet_mask = x["team_fleet_mask"]
        opp_fleet_mask = x["opp_fleet_mask"]
        bs, seq_len, _ = seq.shape
        _, fmap_len, _ = fmap_emb.shape
        _, team_fleet_len, _ = team_fleet_emb.shape
        _, opp_fleet_len, _ = opp_fleet_emb.shape
        seq_mask = torch.full((bs, seq_len), False, dtype=torch.bool, device=seq.device)
        seq_mask[:, fmap_len+2:fmap_len+2+team_fleet_len].copy_(team_fleet_mask)
        seq_mask[:, fmap_len+2+team_fleet_len:].copy_(opp_fleet_mask)
        
        # Encoder
        seq = self.src_ln(seq) # LayerNorm
        seq = seq.permute(1, 0, 2)
        memory = self.encoder(seq, src_key_padding_mask=seq_mask)

        # Decoder (teacher-forcing)
        plan = x["plan"][:, :self.max_tgt_plan_len]
        plan_emb = self.plan_token_embedding(plan)
        plan_pos_emb = torch.arange(plan_emb.shape[1], dtype=torch.long, device=plan_emb.device)
        plan_pos_emb = self.plan_pos_embedding(plan_pos_emb).expand(plan_emb.shape)
        plan_emb = plan_emb + plan_pos_emb

        shipyard_pos_emb = torch.cat([self.pos_x(x["shipyard_pos_x"]), self.pos_y(x["shipyard_pos_y"])], dim=2)

        cls_token = self.tgt_cls_token.expand((bs, 1, self.token_dim))
        cls_token = cls_token + self.team_id(x["team_id"]).expand(cls_token.shape)  # What is the current team?
        cls_token = cls_token + self.step(x["step"]).expand(cls_token.shape) # What is the current time-step?
        cls_token = cls_token + shipyard_pos_emb.expand(cls_token.shape) # Where is the current shipyard?
        cls_token = cls_token + self.team_marker.expand(cls_token.shape) # This shipyard belongs to my team.
        
        # Make tgt_seq
        tgt_seq = torch.cat([cls_token, plan_emb], dim=1)
        # assert tgt_seq.shape[1] == 10, "The tgt_seq's length is incorrect!"

        tgt_attn_mask = self._get_autoregressive_attn_mask(tgt_len=tgt_seq.shape[1], device=tgt_seq.device)

        tgt_seq = self.tgt_ln(tgt_seq) # LayerNorm
        tgt_seq = tgt_seq.permute(1, 0, 2)
        out = self.decoder(
            tgt=tgt_seq, 
            tgt_mask=tgt_attn_mask,
            memory=memory, 
            memory_key_padding_mask=seq_mask
        ).permute(1, 0, 2)
        return out
    
class KoreNet(nn.Module):
    def __init__(self, token_dim, debug):
        super().__init__()
        assert token_dim % 2 == 0, "token_dim should be divisible by 2!"

        self.token_dim = token_dim
        self.debug = debug

        if self.debug:
            torch.set_printoptions(profile="full")

        self.scalar_encoder = ScalarEncoder(1, self.token_dim)
        self.spatial_encoder = SpatialEncoder(token_dim=self.token_dim, input_ch=11, filters=32, layers=12)
        self.fusion_transformer = FusionTransformer(token_dim=self.token_dim)

        self.action_head = nn.Linear(self.token_dim, 3)
        self.spawn_nr_head = nn.Linear(self.token_dim, 11)
        self.launch_nr_head = nn.Sequential(
            nn.Linear(self.token_dim, 1),
            nn.Sigmoid(),
        )
        self.plan_head = nn.Linear(self.token_dim, 18)
    
    def forward(self, x):
        profiler = Timer()
        profile(profiler, "init", self.debug)

        team_info_emb = self.scalar_encoder(x["team_kore"]).unsqueeze(1) # shape (bs, 1, self.token_dim)
        opp_info_emb = self.scalar_encoder(x["opp_kore"]).unsqueeze(1) # shape (bs, 1, self.token_dim)
        profile(profiler, "scalar_encoder", self.debug)
        
        fmap_emb = self.spatial_encoder(x["fmap"])
        bs, ch, w, h = fmap_emb.shape
        fmap_emb = fmap_emb.view(bs, ch, -1).permute(0, 2, 1) # shape (bs, 441, self.token_dim)
        profile(profiler, "spatial_encoder", self.debug)

        out = self.fusion_transformer(
            fmap_emb, 
            team_info_emb,
            x["team_fleet_plan"],
            x["team_fleet_plan_mask"],
            opp_info_emb, 
            x["opp_fleet_plan"],
            x["opp_fleet_plan_mask"],
            x
        )

        profile(profiler, "fusion_transformer", self.debug)
        
        if self.debug:
            assert fmap_emb.isnan().any().item() == False, "fmap_emb contains nan!"
            assert out.isnan().any().item() == False, "out contains nan!"

        return dict(
            action_logit=self.action_head(out[:, 0, :]),
            spawn_nr_logit=self.spawn_nr_head(out[:, 0, :]),
            launch_nr_logit=self.launch_nr_head(out[:, 0, :]),
            plan_logit=self.plan_head(out[:, 1:, :])
        )