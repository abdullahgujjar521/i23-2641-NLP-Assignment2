# i23-2641-NLP-Assignment2

**CS-4063: Natural Language Processing — Assignment 2**  
**Student ID:** I23-2641 | **Section:** DS-C | **FAST NUCES**

---

## Repository Structure

```
i23-2641_Assignment2_DS-C/
├── i23-2641_Assignment2_DS-C.ipynb   # Main notebook (all cells executed)
├── report.pdf                         # 2–3 page report
├── README.md                          # This file
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
├── models/
│   ├── bilstm_pos.pt
│   ├── bilstm_ner.pt
│   └── transformer_cls.pt
└── data/
    ├── pos_train.conll
    ├── pos_test.conll
    ├── ner_train.conll
    └── ner_test.conll
```

---

## Requirements

- Python 3.9+
- PyTorch >= 2.0
- NumPy
- scikit-learn
- matplotlib
- pandas

Install all dependencies:

```bash
pip install torch numpy scikit-learn matplotlib pandas
```

---

## Input Files

Place the following files in the project root before running the notebook:

| File | Used In | Purpose |
|------|---------|---------|
| `cleaned.txt` | All parts | Primary training corpus |
| `raw.txt` | Parts 1 & 2 | Ablation baseline (C2 condition) |
| `Metadata.json` | Part 3 | Topic labels for classification |

---

## How to Reproduce

### Step 1 — Clone the repository

```bash
git clone https://github.com/i23-2641/i23-2641-NLP-Assignment2.git
cd i23-2641-NLP-Assignment2
```

### Step 2 — Install dependencies

```bash
pip install torch numpy scikit-learn matplotlib pandas
```

### Step 3 — Place input files

Copy `cleaned.txt`, `raw.txt`, and `Metadata.json` into the project root directory.

### Step 4 — Run the notebook

Open and run all cells in order:

```bash
jupyter notebook i23-2641_Assignment2_DS-C.ipynb
```

Or run with nbconvert to execute headlessly:

```bash
jupyter nbconvert --to notebook --execute i23-2641_Assignment2_DS-C.ipynb --output i23-2641_Assignment2_DS-C_executed.ipynb
```

---

## Part-by-Part Description

### Part 1 — Word Embeddings

- **TF-IDF:** Term-document matrix (vocab capped at 10,000) with standard IDF weighting. Saved as `embeddings/tfidf_matrix.npy`.
- **PPMI:** Word-word co-occurrence matrix with context window k=5, Positive PMI weighting. Saved as `embeddings/ppmi_matrix.npy`. t-SNE visualisation of top-200 tokens included.
- **Skip-gram Word2Vec:** Trained from scratch with negative sampling (K=10), d=100, k=5, Adam lr=0.001, 5 epochs. Averaged embeddings (V+U)/2 saved as `embeddings/embeddings_w2v.npy`.
- **Four-condition comparison:** C1 (PPMI), C2 (raw.txt), C3 (cleaned.txt), C4 (d=200). MRR reported for each.

### Part 2 — Sequence Labeling

- 500 sentences annotated for POS (12-tag set) and NER (BIO scheme).
- 2-layer bidirectional LSTM with dropout p=0.5, initialised from Word2Vec (C3).
- CRF + Viterbi decoding for NER; linear classifier for POS.
- Models saved as `models/bilstm_pos.pt` and `models/bilstm_ner.pt`.
- Ablation study: unidirectional LSTM, no dropout, random embeddings, softmax vs CRF.

### Part 3 — Transformer Encoder

- Implemented from scratch: scaled dot-product attention, multi-head attention (h=4, d_model=128), position-wise FFN (d_ff=512), sinusoidal PE, 4 Pre-LN encoder blocks, CLS token classifier.
- No `nn.Transformer`, `nn.MultiheadAttention`, or `nn.TransformerEncoder` used.
- Training: AdamW (lr=5e-4, wd=0.01), cosine LR with 50 warmup steps, 20 epochs.
- Model saved as `models/transformer_cls.pt`.

---

## Notes

- All models run on CPU by default. GPU is used automatically if available (`torch.cuda.is_available()`).
- The corpus in this submission is a single BBC Urdu article; all topic labels resolve to **Politics**. This affects stratified splits in Parts 2 and 3.
- CoNLL evaluation files are in `data/` directory.
