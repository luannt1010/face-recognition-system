import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512, m: float = 0.5, s: float = 64.0):
      super().__init__()
      self.m = m
      self.s = s
      self.eps = 1e-8
      self.W = nn.Parameter(torch.empty(num_classes, embedding_dim))
      nn.init.xavier_uniform_(self.W)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
      w_scaled = torch.div(self.W, torch.norm(self.W, p=2, dim=1, keepdim=True).clamp_min(self.eps))
      emb_scaled = torch.div(embeddings, torch.norm(embeddings, p=2, dim=1, keepdim=True).clamp_min(self.eps))

      cosine = torch.mm(emb_scaled, w_scaled.T).clamp(-1+self.eps, 1-self.eps)

      theta = cosine.acos()
      one_hot = torch.zeros_like(cosine)
      one_hot.scatter_(1, labels.view(-1, 1), 1.0)

      # If j != y
      theta_diff = cosine * (1 - one_hot)
      # If j == y
      theta_similar = torch.cos(theta + self.m) * one_hot

      logits = self.s * (theta_diff + theta_similar)
      loss = F.cross_entropy(logits, labels)
      return loss

class AdaFaceLoss(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512, m: float = 0.4, s: float = 64.0, t_alpha: float = 0.01):
      super().__init__()
      self.m = m
      self.s = s
      self.t_alpha = t_alpha
      self.h = 0.333
      self.eps = 1e-3
      self.W = nn.Parameter(torch.empty(num_classes, embedding_dim))
      self.W.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
      # nn.init.xavier_uniform_(self.W)

      # Running stats để normalize feature norm → norm̂ ∈ [-1, 1]
      self.register_buffer("running_mean", torch.ones(1) * 20.0)
      self.register_buffer("running_std", torch.ones(1) * 100)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
      # norm feature
      z_normed = torch.norm(embeddings, p=2, dim=1, keepdim=True)
      safe_norm = (z_normed.clip(min=0.001, max=100)).clone().detach()
      if self.training:
        with torch.no_grad():
          mean = safe_norm.mean().detach()
          std = safe_norm.std(unbiased=False).detach()
          self.running_mean.mul_(1 - self.t_alpha).add_(mean * self.t_alpha)
          self.running_std.mul_(1 - self.t_alpha).add_(std * self.t_alpha)

      embeddings = torch.div(embeddings, z_normed.clamp_min(self.eps))
      # scale feature
      z_norm_hat = (z_normed - self.running_mean) / (self.running_std+self.eps)
      z_norm_hat = (z_norm_hat * self.h).clip(-1, 1)

      # norm weight
      weight_norm = torch.div(self.W, torch.norm(self.W, dim=1, keepdim=True).clamp_min(self.eps))

      # calc angle of feature and weight
      cosine = (torch.mm(embeddings, weight_norm.T)).clip(-1+self.eps, 1-self.eps)

      # calc g_angular
      m_arc = torch.zeros_like(cosine)
      m_arc.scatter_(1, labels.view(-1, 1), 1.0)
      g_angular = -self.m * z_norm_hat
      m_arc = m_arc * g_angular
      theta = cosine.acos()
      theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
      cosine = torch.cos(theta_m)

      # calc g_additive
      m_cos = torch.zeros_like(cosine)
      m_cos.scatter_(1, labels.view(-1, 1), 1.0)
      g_add = self.m * z_norm_hat + self.m
      m_cos = m_cos * g_add
      cosine = cosine - m_cos

      logits = cosine * self.s
      loss = F.cross_entropy(logits, labels)
      return loss

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2):
      super().__init__()
      self.margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
      labels = labels.view(-1)
      distances = torch.cdist(embeddings, embeddings, p=2)

      same_class = labels.unsqueeze(0).eq(labels.unsqueeze(1))
      identity = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
      positive_mask = same_class & ~identity
      negative_mask = ~same_class

      valid_anchors = positive_mask.any(dim=1) & negative_mask.any(dim=1)
      if not valid_anchors.any():
        return embeddings.sum() * 0.0

      hardest_positive = distances.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
      hardest_negative = distances.masked_fill(~negative_mask, float("inf")).min(dim=1).values

      losses = F.relu(hardest_positive - hardest_negative + self.margin)
      return losses[valid_anchors].mean()



