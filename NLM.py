import torch
from torch import nn
from torch.nn import functional as F


class BasicNeuralLanguageModel(nn.Module):
    def __init__(
        self,
        vocabSize=None,
        contextSize=None,
        embeddingDim=None,
        hiddenDim=None,
        vocab_size=None,
        context_size=None,
        embedding_dim=None,
        hidden_dim=None,
    ):
        super().__init__()
        if vocabSize is None:
            vocabSize = vocab_size
        if contextSize is None:
            contextSize = context_size
        if embeddingDim is None:
            embeddingDim = embedding_dim
        if hiddenDim is None:
            hiddenDim = hidden_dim

        if vocabSize is None or contextSize is None:
            raise TypeError("vocabSize and contextSize are required")

        if embeddingDim is None:
            embeddingDim = 64
        if hiddenDim is None:
            hiddenDim = 128

        self.vocabSize = int(vocabSize)
        self.contextSize = int(contextSize)
        self.embeddingDim = int(embeddingDim)
        self.hiddenDim = int(hiddenDim)

        self.embedding = nn.Embedding(self.vocabSize, self.embeddingDim)
        self.fc1 = nn.Linear(self.contextSize * self.embeddingDim, self.hiddenDim)
        self.fc2 = nn.Linear(self.hiddenDim, self.vocabSize)

    def forward(self, contexts: torch.Tensor) -> torch.Tensor:
        x = self.embedding(contexts)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)

    def loss(self, contexts: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = self(contexts)
        return F.cross_entropy(logits, targets)
