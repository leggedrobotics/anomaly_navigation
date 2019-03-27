import torch
import torch.nn as nn
import torch.nn.functional as F




class STNetwork(nn.Module):
  def __init__(self, in_dim, mid_dim):
    super().__init__()

    self.layers = nn.Sequential(
              nn.Conv2d(in_dim, mid_dim, 1, bias=True),
              nn.LeakyReLU(),
              nn.Conv2d(mid_dim, mid_dim, 1, bias=True),
              nn.LeakyReLU(),
              nn.Conv2d(mid_dim, 2*in_dim, 1, bias=True),
              nn.Tanh()
            )
    for m in self.modules():
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)

  def forward(self, x):
    x = self.layers(x)
    return x.chunk(2, dim=1)



class RealNVPBlock(nn.Module):
  def __init__(self, in_dim, mid_dim):
    super().__init__()
    self.st_net = STNetwork(in_dim, mid_dim)

  def forward(self, x, mask, inv_mask):
    # Split channels.
    x_fix = F.conv2d(x, inv_mask)
    x_change = F.conv2d(x, mask)

    # Compute scaling and translation.
    s, t = self.st_net(x_fix)
    x_change = x_change * s.exp() + t

    # Write changes back to channels.
    mask_nochange = inv_mask.sum(dim=0)
    mask_change = mask.sum(dim=0)
    x = x * mask_nochange
    x_change_long = F.conv2d(x_change, mask.transpose(0, 1))
    x = x + x_change_long

    return x, s.sum(dim=1, keepdim=True)



class RealNVP(nn.Module):
  def __init__(self, in_dim, mid_dim):
    super().__init__()
    eye = torch.eye(in_dim//2)
    zeros = torch.zeros(in_dim//2, in_dim//2)
    a1 = torch.cat((eye, zeros), dim=1).unsqueeze_(-1).unsqueeze_(-1)
    a2 = torch.cat((zeros, eye), dim=1).unsqueeze_(-1).unsqueeze_(-1)
    b1 = torch.stack((eye, zeros), dim=-1).reshape(in_dim//2, in_dim).unsqueeze_(-1).unsqueeze_(-1)
    b2 = torch.stack((zeros, eye), dim=-1).reshape(in_dim//2, in_dim).unsqueeze_(-1).unsqueeze_(-1)
    self.masks = [(a1, a2),
                  (a2, a1),
                  (b1, b2),
                  (b2, b1),
                  (a1, a2),
                  (a2, a1),
                  (b1, b2),
                  (b2, b1)]
    self.nets = nn.ModuleList([RealNVPBlock(in_dim//2, mid_dim) for _ in self.masks])

  def to(self, device):
    super().to(device)
    self.masks = [(mask.to(device), inv_mask.to(device)) for mask, inv_mask in self.masks]
    return self

  def forward(self, x):
    log_det_J = torch.zeros((x.shape[0], 1, x.shape[2], x.shape[3]), device=x.device)
    for net, (mask, inv_mask) in zip(self.nets, self.masks):
      x, cur_log_det_J = net(x, mask, inv_mask)
      log_det_J = log_det_J + cur_log_det_J
    return x, log_det_J



class EncoderRealNVP(nn.Module):
  def __init__(self, encoder, nvp):
    super().__init__()
    self.encoder = encoder
    self.nvp = nvp
    self.rep_dim = encoder.rep_dim

  def to(self, device):
    super().to(device)
    self.encoder.to(device)
    self.nvp.to(device)
    return self

  def forward(self, x):
    x = self.encoder(x)
    return self.nvp(x)


