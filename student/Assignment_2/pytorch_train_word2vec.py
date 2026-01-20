import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Hyperparameters
EMBEDDING_DIM = 100
BATCH_SIZE = 128
EPOCHS = 25
LEARNING_RATE = 0.01
NEGATIVE_SAMPLES = 5  # Number of negative samples per positive

# Custom Dataset for Skip-gram
class SkipGramDataset(Dataset):
    def __init__(self, skipgram_df):
        self.df = skipgram_df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        center = int(row["center"])
        context = int(row["context"])
        return torch.tensor(center, dtype=torch.long), torch.tensor(context, dtype=torch.long)


# Simple Skip-gram Module
class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.in_embed = nn.Embedding(vocab_size, embedding_dim)
        self.out_embed = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, center_words, context_words):
        v = self.in_embed(center_words)
        u = self.out_embed(context_words)
        return (v * u).sum(dim=1)


# Load processed data
with open("processed_data.pkl", "rb") as f:
    data = pickle.load(f)

skipgram_df = data["skipgram_df"]
word2idx = data["word2idx"]
idx2word = data["idx2word"]
vocab_size = len(word2idx)

# Precompute negative sampling distribution below (use precomputed counter)
counts = torch.zeros(vocab_size, dtype=torch.float)
counter = data["counter"]  # word -> count
for word, c in counter.items():
    counts[word2idx[word]] = float(c)

neg_dist = counts.pow(0.75)
neg_dist = neg_dist / neg_dist.sum()

# Device selection: CUDA > MPS > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Dataset and DataLoader (shuffle required)
dataset = SkipGramDataset(skipgram_df)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model, Loss, Optimizer
model = Word2Vec(vocab_size, EMBEDDING_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()

def make_targets(center, context, vocab_size):
    batch_size = center.size(0)

    neg_samples = torch.multinomial(
        neg_dist.to(center.device),
        num_samples=batch_size * NEGATIVE_SAMPLES,
        replacement=True
    ).view(batch_size, NEGATIVE_SAMPLES)

    # make sure negatives do not include the positive context word
    mask = (neg_samples == context.unsqueeze(1))
    while mask.any():
        neg_samples[mask] = torch.multinomial(
            neg_dist.to(center.device),
            num_samples=int(mask.sum().item()),
            replacement=True
        )
        mask = (neg_samples == context.unsqueeze(1))

    return neg_samples

# Training loop
for epoch in range(EPOCHS):
    for center, context in dataloader:
        center = center.to(device)
        context = context.to(device)

        optimizer.zero_grad()

        pos_logits = model(center, context)

        neg_samples = make_targets(center, context, vocab_size)

        v = model.in_embed(center)                    # (B, D)
        u_neg = model.out_embed(neg_samples)          # (B, K, D)
        neg_logits = (u_neg * v.unsqueeze(1)).sum(2)  # (B, K)

        logits = torch.cat([pos_logits, neg_logits.view(-1)], dim=0)

        labels = torch.cat([
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits).view(-1)
        ], dim=0)

        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

# Save embeddings and mappings (input embeddings only)
embeddings = model.in_embed.weight.detach().cpu().numpy()
with open('word2vec_embeddings.pkl', 'wb') as f:
    pickle.dump({'embeddings': embeddings, 'word2idx': word2idx, 'idx2word': idx2word}, f)

print("Embeddings saved to word2vec_embeddings.pkl")
