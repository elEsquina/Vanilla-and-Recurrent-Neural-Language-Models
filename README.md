# Neural Language Models 

Layout:
- Datasets: `data/News_Category_Dataset_v2.json`, `data/AmazonReviews/train.csv`
- Models: `NLM.py`, `ReccurentNLM.py`
- NLP pipeline: `nlp_pipeline.py`
- Comparison notebook: `Testing.ipynb`
- LaTeX report: `latex/main.tex`

In your code or your jupyter:
```python
from NLM import BasicNeuralLanguageModel
from ReccurentNLM import RNNNeuralLanguageModel

basic = BasicNeuralLanguageModel(vocabSize=1200, contextSize=4, embeddingDim=32, hiddenDim=64)
rnn = RNNNeuralLanguageModel(vocabSize=1200, embeddingDim=32, hiddenDim=64)
```

Notebook:
- `Testing.ipynb` trains the Basic (fixed-context) model and the RNN model on both datasets, saves plots into `latex/figs/`, and writes `latex/results.json` and `latex/results_table.tex`.

NLP pipeline utilities:
```python
from nlp_pipeline import Tokenizer, Vocabulary, makeNgramXY, makeRnnXY
```
