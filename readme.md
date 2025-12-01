
# MovieLens & Amazon Books Recommender System Notebooks

This repository contains preprocessing and baseline notebooks for both the MovieLens 1M and Amazon Books datasets. It includes candidate pool generation, ALS and NCF model training, and evaluation for recommender system and LLM baselines.

## Workflow Overview

### MovieLens 1M
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
   - Outputs splits and ID maps to `movielens_dataset/splits/`.
4. **Sparse Matrix Construction**
   - Builds a user × item sparse matrix from the train split for ALS and kNN.
5. **Pool Generation**
   - For each user, builds a candidate pool by merging:
     - Most popular unseen items (train-only)
     - ALS recommendations
     - Item-item kNN neighbors
   - Dedupes, drops seen items, and caps pool size.
   - Saves pools to `movielens_dataset/candidates/`.
6. **Coverage Calculation**
   - Computes what fraction of users have their held-out item in their candidate pool.
7. **ALS Model Training**
   - Trains ALS (from `implicit`) on the train matrix (with BM25 weighting).
   - Embeddings are used for scoring candidate pools.
8. **Evaluation**
   - Computes HR@10 and NDCG@10 for validation and test sets.
9. **Hyperparameter Tuning**
   - Grid search over ALS parameters, reporting best results.

### Amazon Books
1. **Data Preprocessing**
   - Loads and cleans Amazon Books ratings and metadata.
   - Filters users/items, handles sparsity, and creates train/val/test splits similar to MovieLens.
2. **Candidate Pool Generation**
   - Builds candidate pools for each user using popularity, ALS, and kNN methods.
3. **ALS Model Training & Evaluation**
   - Trains ALS on the train split and evaluates HR@10, NDCG@10.
4. **NCF Baseline**
   - Trains Neural Collaborative Filtering (NCF) baseline on the Amazon Books data.
   - Evaluates NCF performance on validation and test sets.

## File Structure

```
movielens_recsys.ipynb             # MovieLens preprocessing, candidate pools, ALS baseline
amazonbooks.ipynb                  # Amazon Books preprocessing, candidate pools, ALS baseline
amazonbooks_ncf_baseline.ipynb     # Amazon Books NCF baseline training & evaluation
movielens_dataset/
    movies.csv                     # MovieLens movies data
    ratings.csv                    # MovieLens ratings data
    ratings_limited.csv            # Ratings capped at 5/user
    splits/                        # Train/val/test splits & ID maps
    candidates/                    # Candidate pools for val/test users
.gitignore
readme.md                          # This file
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

3. **Run the notebooks**
   - Open any notebook (`movielens_recsys.ipynb`, `amazonbooks.ipynb`, `amazonbooks_ncf_baseline.ipynb`) in VS Code or Jupyter.
   - Execute cells in order.

## Outputs

- **Splits**: Train/val/test splits and ID maps in `movielens_dataset/splits/`
- **Candidate Pools**: For val/test users in `movielens_dataset/candidates/`
- **ALS Model Results**: HR@10 and NDCG@10 for validation and test sets
- **NCF Baseline Results**: HR@10 and NDCG@10 for Amazon Books validation and test sets

## Notes

- All candidate pools and splits are saved as Parquet and CSV for compatibility.
- The notebooks are designed for reproducibility and easy integration with LLM-based baselines.