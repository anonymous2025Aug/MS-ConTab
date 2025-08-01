
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pytorch_tabnet.tab_network import TabNetEncoder
import torch.nn as nn
import random

# CONFIGURATION
SEED = 42
BATCH_SIZE = 8
LATENT_DIM = 128  # TabNet encoder output size
PROJ_DIM = 64     # Output size of projection head (dimension of contrastive space)
LR = 1e-3
EPOCHS = 100
TEMPERATURE = 0.5
GENE_FILE = 'gene1_count.xlsx'
CHROM_FILE = 'chrom1_count.xlsx'

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cpu')

# DATA LOADING
gene_xl = pd.ExcelFile(GENE_FILE)
chrom_xl = pd.ExcelFile(CHROM_FILE)
gene_sheets = [s for s in gene_xl.sheet_names if 'Frequently' not in s]
chrom_sheets = chrom_xl.sheet_names
cancers = sorted(set(gene_sheets) & set(chrom_sheets))

gene_feats, chrom_feats = [], []
for c in cancers:
    df_g = gene_xl.parse(c).select_dtypes(include=np.number)
    df_c = chrom_xl.parse(c).select_dtypes(include=np.number)
    gene_feats.append(df_g.values.flatten())
    chrom_feats.append(df_c.values.flatten())

gene_X = np.stack(gene_feats)
chrom_X = np.stack(chrom_feats)

class MultiViewCancerDataset(Dataset):
    def __init__(self, gene, chrom):
        self.gene = torch.tensor(gene, dtype=torch.float32)
        self.chrom = torch.tensor(chrom, dtype=torch.float32)
    def __len__(self):
        return len(self.gene)
    def __getitem__(self, idx):
        return self.gene[idx], self.chrom[idx]

dataloader = DataLoader(MultiViewCancerDataset(gene_X, chrom_X),
                        batch_size=BATCH_SIZE, shuffle=True)

# TabNet Encoders
gene_enc = TabNetEncoder(
    input_dim=gene_X.shape[1], output_dim=LATENT_DIM,
    n_d=128, n_a=128, n_steps=3, gamma=1.5
).to(device)
chrom_enc = TabNetEncoder(
    input_dim=chrom_X.shape[1], output_dim=LATENT_DIM,
    n_d=128, n_a=128, n_steps=3, gamma=1.5
).to(device)

# Projection Heads
class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    def forward(self, x):
        return self.proj(x)

proj_gene = ProjectionHead(LATENT_DIM, PROJ_DIM).to(device)
proj_chrom = ProjectionHead(LATENT_DIM, PROJ_DIM).to(device)

def get_tensor(x):
    if isinstance(x, (tuple, list)):
        x = x[0]
    if isinstance(x, (tuple, list)):
        x = x[0]
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"TabNet output is not tensor, got {type(x)}: {x}")
    return x

def nt_xent_loss(z1, z2, temp=TEMPERATURE):
    # z1, z2: (batch, PROJ_DIM)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # (2N, PROJ_DIM)
    sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    mask = torch.eye(2*N, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    labels = torch.arange(N, device=z.device)
    labels = torch.cat([labels + N, labels], dim=0)
    return F.cross_entropy(sim, labels)

optimizer = torch.optim.Adam(
    list(gene_enc.parameters()) + list(chrom_enc.parameters()) +
    list(proj_gene.parameters()) + list(proj_chrom.parameters()), lr=LR
)

# TRAINING LOOP
for epoch in range(1, EPOCHS+1):
    gene_enc.train(); chrom_enc.train(); proj_gene.train(); proj_chrom.train()
    total_loss = 0.0
    for gv, cv in dataloader:
        z_g = get_tensor(gene_enc(gv))
        z_c = get_tensor(chrom_enc(cv))
        z_g = proj_gene(z_g)
        z_c = proj_chrom(z_c)
        loss = nt_xent_loss(z_g, z_c)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}/{EPOCHS} Loss: {total_loss/len(dataloader):.4f}")

# SAVE EMBEDDINGS (in the aligned space)
with torch.no_grad():
    z_g = proj_gene(get_tensor(gene_enc(torch.tensor(gene_X, dtype=torch.float32))))
    z_c = proj_chrom(get_tensor(chrom_enc(torch.tensor(chrom_X, dtype=torch.float32))))
    embeds = ((z_g + z_c) / 2).cpu().numpy()
    np.save("cancer_embeddings_tabnet_proj.npy", embeds)
print("Saved embeddings to cancer_embeddings_tabnet_proj.npy")
