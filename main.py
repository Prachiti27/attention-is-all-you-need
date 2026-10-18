import torch
import torch.nn.functional as F
import math

# Token representations: 3 tokens, each with 4 features
X = torch.randn(3, 4)

# Learnable projection matrices
Wq = torch.randn(4, 4)
Wk = torch.randn(4, 4)
Wv = torch.randn(4, 4)

# Step 1: Compute Query, Key, Value
Q = X @ Wq
K = X @ Wk
V = X @ Wv

# Step 2: Compute attention scores
scores = Q @ K.T / math.sqrt(K.shape[-1])

# Step 3: Normalize scores
attention = F.softmax(scores, dim=-1)

# Step 4: Compute contextual representations
output = attention @ V

print("Attention weights:\n", attention)
print("Output representations:\n", output)