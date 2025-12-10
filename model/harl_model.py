import os.path

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch_geometric.nn import GATConv, GraphConv, GCNConv, AGNNConv, EdgeConv
from torch_geometric.data import Data as gdata
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected

gnn_type_2_func = {
    "GraphConv": GraphConv,
    "GATConv": GATConv,
    "GCNConv": GCNConv,
    "AGNNConv": AGNNConv,
    "EdgeConv": EdgeConv
}

class edge_regression(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.linear1 = nn.Linear(self.channel, self.channel)
        self.linear2 = nn.Linear(self.channel, self.channel)
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, f1, f2):
        bs = f1.shape[0]
        diff = f1.unsqueeze(-1) - f2.unsqueeze(1)
        diff = diff.view(-1, self.channel)
        diff = self.linear1(diff)
        diff = self.relu(diff)
        diff = self.linear2(diff)
        # diff = self.sigmoid(diff)
        diff = self.softmax(diff)
        diff = diff.view(bs, -1, self.channel)

        return diff
    

class HARL(nn.Module):
    def __init__(self, gnn_layers, eye_encoder, face_encoder):
        super().__init__()
        self.eye_model = eye_encoder
        self.face_model = face_encoder
        self.gnn_layers = gnn_layers
        self.EyeBranch_num_feat = self.eye_model.num_features
        self.FaceBranch_num_feat = self.face_model.num_features

        self.regression_edges = nn.ModuleList()
        self.aggregation_layers = nn.ModuleList()

        for _ in range(self.gnn_layers):
            self.regression_edges.append(edge_regression(channel=self.EyeBranch_num_feat))
            self.aggregation_layers.append(GCNConv(1, 1))

        self.mlp = nn.Sequential(
            nn.Linear(self.EyeBranch_num_feat * 2 + self.FaceBranch_num_feat, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
        self.inter_num_feat = self.EyeBranch_num_feat * 2 + self.FaceBranch_num_feat
        self.freeze_()

    def freeze_(self):
        for _, param in self.face_model.named_parameters():
            param.requires_grad = False
    

    def batch_graph(self, f1, f2, am):
        bs = f1.shape[0]
        data_list = []

        for i in range(bs):
            adj = am[i]
            feat1 = f1[i]
            feat2 = f2[i]
            edge_index, edge_weight = self.affinity2edge(adj)
            node = torch.cat([feat1, feat2], dim=0)
            data_list.append(gdata(x=node, edge_index=edge_index))
        graph = Batch.from_data_list(data_list)
        return graph

    def affinity2edge(self, am, k=1):
        H, W = am.shape
        mask, _ = torch.max(am, dim=-1, keepdim=True)
        # mask = mask.unsqueeze(-1)
        mask = (am == mask).float()

        edge_index = torch.where(mask > 0)
        edge_index = torch.stack(edge_index, dim=0)

        edge_index[1, :] = edge_index[1, :] + H
        edge_index[0, :] = edge_index[0, :]

        edge_weight = am[mask.bool()].float()
        edge_index, edge_weight = to_undirected(edge_index, edge_weight)

        return edge_index, edge_weight

    def forward_train(self, sample, sample_hq):
        
        ## hybrid-domain adaptation

        feat_left = self.eye_model(sample['left'], requires_out=False)
        feat_right = self.eye_model(sample['right'], requires_out=False)
        feat_hq, gaze_hq = self.eye_model(sample_hq['hq_mono'], requires_out=True)

        feat_pose = self.face_model(sample['face'])

        feat_bino = torch.cat([feat_left, feat_right], dim=1)

        bs = feat_bino.shape[0]
        begin_dim = 2 * self.EyeBranch_num_feat

        ## graph-based feature fusion

        for i in range(self.gnn_layers):
            if i == 0:
                affinity_matrix = self.regression_edges[i](feat_bino, feat_pose)
                graph = self.batch_graph(feat_bino.unsqueeze(-1), feat_pose.unsqueeze(-1), affinity_matrix)
                graph.x = self.aggregation_layers[i](graph.x, graph.edge_index)
            else:
                feat_inter = graph.x
                feat_inter = feat_inter.reshape(bs, self.inter_num_feat, 1)
                feat_bino = feat_inter[:, :begin_dim, :]
                feat_pose = feat_inter[:, begin_dim:, :]
                affinity_matrix = self.regression_edges[i](feat_bino.squeeze(-1), feat_pose.squeeze(-1))
                graph = self.batch_graph(feat_bino, feat_pose, affinity_matrix)
                graph.x = self.aggregation_layers[i](graph.x, graph.edge_index)
        
        feat_gaze = graph.x
        feat_gaze = feat_gaze.reshape(bs, self.inter_num_feat, 1).squeeze(-1)
        face_gaze = self.mlp(feat_gaze)

        results = {
            'face_gaze':face_gaze,
            'hq_gaze': gaze_hq,
            'feat_hq': feat_hq,
            'feat_left': feat_left,
            'feat_right': feat_right,
            'gt_hq_gaze': sample_hq['hq_gaze2d'],
            'gt_face_gaze': sample['face_gaze2d']
        }

        return results

    def forward_test(self, sample):

        feat_left = self.eye_model(sample['left'], requires_out=False)
        feat_right = self.eye_model(sample['right'], requires_out=False)
        feat_pose = self.face_model(sample['face'])

        feat_bino = torch.cat([feat_left, feat_right], dim=1)
        bs = feat_bino.shape[0]
        begin_dim = 2 * self.EyeBranch_num_feat
        for i in range(self.gnn_layers):
            if i == 0:
                affinity_matrix = self.regression_edges[i](feat_bino, feat_pose)
                graph = self.batch_graph(feat_bino.unsqueeze(-1), feat_pose.unsqueeze(-1), affinity_matrix)
                graph.x = self.aggregation_layers[i](graph.x, graph.edge_index)
            else:
                feat_inter = graph.x
                feat_inter = feat_inter.reshape(bs, self.inter_num_feat, 1)
                feat_bino = feat_inter[:, :begin_dim, :]
                feat_pose = feat_inter[:, begin_dim:, :]
                affinity_matrix = self.regression_edges[i](feat_bino.squeeze(-1), feat_pose.squeeze(-1))
                graph = self.batch_graph(feat_bino, feat_pose, affinity_matrix)
                graph.x = self.aggregation_layers[i](graph.x, graph.edge_index)
        
        feat_gaze = graph.x
        feat_gaze = feat_gaze.reshape(bs, self.inter_num_feat, 1).squeeze(-1)
        face_gaze = self.mlp(feat_gaze)

        results = {
            'face_gaze': face_gaze,
            'gt_face_gaze': sample['face_gaze2d']
        }

        return results

    def forward(self, sample, sample_hq=None):
        if self.training:
            return self.forward_train(sample, sample_hq)
        else:
            return self.forward_test(sample)