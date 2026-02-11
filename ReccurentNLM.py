import torch
from torch import nn
from torch.nn import functional as F


class RNNNeuralLanguageModel(nn.Module):
    def __init__(
        self,
        vocabSize=None,
        embeddingDim=None,
        hiddenDim=None,
        numLayers=None,
        vocab_size=None,
        embedding_dim=None,
        hidden_dim=None,
        num_layers=None,
    ):
        super().__init__()
        if vocabSize is None:
            vocabSize = vocab_size
        if embeddingDim is None:
            embeddingDim = embedding_dim
        if hiddenDim is None:
            hiddenDim = hidden_dim
        if numLayers is None:
            numLayers = num_layers

        if vocabSize is None:
            raise TypeError("vocabSize is required")

        if embeddingDim is None:
            embeddingDim = 64
        if hiddenDim is None:
            hiddenDim = 128
        if numLayers is None:
            numLayers = 1

        self.vocabSize = int(vocabSize)
        self.embeddingDim = int(embeddingDim)
        self.hiddenDim = int(hiddenDim)
        self.numLayers = int(numLayers)

        self.embedding = nn.Embedding(self.vocabSize, self.embeddingDim)
        self.rnn = nn.RNN(
            input_size=self.embeddingDim,
            hidden_size=self.hiddenDim,
            num_layers=self.numLayers,
            nonlinearity="tanh",
            batch_first=True,
        )
        self.fc = nn.Linear(self.hiddenDim, self.vocabSize)

    def forward(self, tokens: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(tokens)
        out, hn = self.rnn(x, h0)
        logits = self.fc(out)
        return logits, hn

    def loss(self, tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits, _ = self(tokens)
        return F.cross_entropy(logits.reshape(-1, self.vocabSize), targets.reshape(-1))
