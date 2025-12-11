
# Evaluating LLMs as Alternatives to Traditional RecSys on MovieLens & Amazon Books

This repository contains preprocessing, baseline recommender models, and LLM experiments for both the **MovieLens 1M** and **Amazon Books** datasets. It includes:

- Candidate pool generation
- ALS and NeuMF/NCF baselines
- LLM-based rerankers via zero-shot prompting and LoRA fine-tuning

The goal is to compare **traditional recommenders** against **LLM-based rankers** under the same candidate-pool evaluation setup (HR@K / NDCG@K).

## Workflow Overview

### MovieLens 1M (RecSys Baselines)

1. **Data Conversion**
   - Converts `ratings.dat` and `movies.dat` to CSV for easier handling.

2. **Ratings Limiting**
   - Limits each user to their 5 most recent ratings for fair train/test splits.

3. **Time-Aware Leave-One-Out Split**
   - Splits ratings per user:
     - Last rating → **test**
     - Second last → **val** (if exists)
     - The rest → **train**
   - Only users with ≥4 positive ratings (≥3.0) are kept.
   - Outputs splits and ID maps to `movielens/data/dataset/splits/`.

4. **Sparse Matrix Construction**
   - Builds a user × item sparse matrix from the train split for ALS and kNN.

5. **Candidate Pool Generation**
   - For each user, builds a candidate pool by merging:
     - Most popular unseen items (train-only)
     - ALS recommendations
     - Item–item kNN neighbors
   - Deduplicates, drops already-seen items, and caps pool size to 50.
   - Saves pools to `movielens/data/dataset/candidates/`.

6. **Coverage Calculation**
   - Computes the fraction of users whose held-out item is present in their candidate pool.

7. **ALS Model Training**
   - Trains ALS (via `implicit`) on the train matrix (with BM25 weighting).
   - Uses learned embeddings to score candidates.

8. **Evaluation**
   - Computes HR@K and NDCG@K for validation and test sets.
   - Used as the main traditional baseline for MovieLens.

---

### Amazon Books (RecSys Baselines)

1. **Data Preprocessing**
   - Loads and cleans Amazon Books ratings and metadata.
   - Filters users/items, handles sparsity, and creates MovieLens-style train/val/test splits.

2. **Candidate Pool Generation**
   - Builds candidate pools for each user using popularity, ALS, and kNN.

3. **ALS Model Training & Evaluation**
   - Trains ALS on the train split and evaluates HR@K and NDCG@K.

4. **NCF / NeuMF Baseline**
   - Trains a Neural Collaborative Filtering (NeuMF-style) model.
   - Evaluates NeuMF performance on validation and test candidate pools.

---

## LLMs

### MovieLens LLM Notebook  
**`movielens/notebooks/movielens_llm_zeroshot+finetune.ipynb`**

This notebook implements **LLM-based rerankers** on top of the MovieLens candidate pools:

- **Zero-shot LLM reranker**
  - Uses `Llama-3.2-3B-Instruct-bnb-4bit` loaded via **Unsloth**.
  - For each user:
    - Builds a chat-style prompt containing the last 3 movies in their history and the 50-item candidate pool (IDs, titles, genres).
    - Asks the LLM to return a JSON list of candidate IDs in ranked order.
  - Parses the output and evaluates HR@K / NDCG@K on the same candidate pools used by ALS.

- **CTR-style fine-tuned LLM (LoRA)**
  - Builds a **binary CTR dataset** from the MovieLens train split:
    - Each example = (user history, single candidate movie, label ∈ {YES, NO}).
  - Converts examples into instruction-style prompts:
    - “Given this history and candidate, will the user watch it next? Answer YES or NO.”
  - Fine-tunes `Llama-3.2-3B-Instruct-bnb-4bit` with **LoRA** (via Unsloth + TRL `SFTTrainer`).
  - At inference:
    - Scores each candidate by p(YES | history, candidate) from the YES/NO logits.
    - Reranks the same 50-item candidate pools and recomputes HR@K / NDCG@K.
  - Includes logging of prompts, rankings, and a small learning-curve plot for the LoRA adapters.

This notebook is the main place to look for **LLM vs ALS** experiments on MovieLens.

---

### Amazon Books LLM Notebook  
**`amazonbooks/notebooks/ECE1508_Zero_Shot_+_Fine_Tuning.ipynb`**

This notebook mirrors the MovieLens LLM setup, but on **Amazon Books**:

- **Zero-shot LLM baseline**
  - Uses an instruction-tuned LLM (via Unsloth) to rerank AmazonBooks candidate pools.
  - Prompts include the user’s recent interaction history (books) and a candidate book list with metadata.
  - Evaluates HR@K / NDCG@K on precomputed candidate pools.

- **LoRA fine-tuning for CTR**
  - Constructs a binary dataset of (history, candidate, label) from the AmazonBooks training interactions.
  - Fine-tunes the LLM with LoRA to predict YES/NO for each user–book pair.
  - Uses the resulting p(YES | history, candidate) scores to rerank candidates and compare against NeuMF/ALS.

This notebook provides the **LLM counterpart to the AmazonBooks NeuMF baseline**, enabling MovieLens–AmazonBooks cross-dataset comparisons for LLM-based ranking.

---

## File Structure

```
```text
amazonbooks/
    dataset/
        ...                        # Preprocessed Amazon Books data, splits, candidates
    notebooks/
        amazonbooks.ipynb                  # Amazon Books preprocessing, candidate pools, ALS / basic baselines
        amazonbooks_ncf_baseline2.ipynb    # Amazon Books NeuMF / NCF baseline training and evaluation
        ECE1508_Zero_Shot_+_Fine_Tuning.ipynb   # LLM zero-shot + LoRA fine-tuning on Amazon Books

movielens/
    data/
        ctr_datasets/              # CTR-style JSONL datasets (train/val/all) for LLM fine-tuning
        dataset/                   # Raw MovieLens data, splits, candidate pools
    logs/
        ...                        # JSON logs from LLM runs (prompts, rankings, metrics)
    models/
        ...                        # Saved LoRA adapters and checkpoints for LLM fine-tuning
    notebooks/
        movielens_recsys.ipynb                 # MovieLens preprocessing, candidate pools, ALS baseline
        movielens_llm_zeroshot+finetune.ipynb  # MovieLens LLM zero-shot + CTR-style fine-tuning experiments

.gitignore
.gitattributes
readme.md
```

## Setup

1. **Create a Python virtual environment**
   ```sh
   python -m venv .venv
   # Activate:
   # Windows (PowerShell): .\.venv\Scripts\Activate.ps1
   # Mac/Linux: source .venv/bin/activate
   ```

2. **Install dependencies**
   ```sh
   python -m pip install --upgrade pip
   python -m pip install pandas numpy pyarrow scipy scikit-learn tqdm implicit torch
   ```

2. **Extra dependencies for LLM notebooks (already included in notebook file)**
   ```sh
   pip install transformers accelerate peft datasets trl unsloth
   ```

3. **Run the notebooks**
   - Open any notebook (e.g., movielens_recsys.ipynb, movielens_llm_zeroshot+finetune.ipynb, amazonbooks.ipynb, ECE1508_Zero_Shot_+_Fine_Tuning.ipynb) in Colab.
   - Execute cells in order.

## Outputs

- **Splits**: Train/val/test splits and ID maps in `movielens/data/dataset/splits/` and `amazonbooks/dataset/splits/`
- **Candidate Pools**: Candidate lists for val/test users: in `movielens/data/dataset/candidates/` and `amazonbooks/dataset/candidates_subset100/`
- **RecSys Baseline Results**: ALS and NCF HR@K / NDCG@K stored in the corresponding notebooks.
- **LLM Logs & Models**: Prompt + ranking logs: `movielens/logs/llama32_3b_*.json`, LoRA adapter: `movielens/models/llama3.2_3b_movielens_ctr_lora/`