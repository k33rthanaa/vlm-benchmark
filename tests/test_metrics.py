import pytest
import numpy as np
from vlm_benchmark.metrics import Metrics

def test_accuracy():
    preds = [1, 0, 1, 1]
    refs = [1, 0, 0, 1]
    assert Metrics.accuracy(preds, refs) == 0.75

def test_recall_at_k():
    # 3 queries, 3 candidates
    # Perfect retrieval
    sim_matrix = np.array([
        [0.9, 0.1, 0.0], # Q0 -> Cand0 is top
        [0.2, 0.8, 0.0], # Q1 -> Cand1 is top
        [0.1, 0.2, 0.7]  # Q2 -> Cand2 is top
    ])
    ground_truth = [0, 1, 2]
    
    metrics = Metrics.recall_at_k(sim_matrix, ground_truth, k_values=[1])
    assert metrics['recall_at_1'] == 1.0

    # Imperfect retrieval
    sim_matrix_bad = np.array([
        [0.1, 0.9, 0.0], # Q0 -> Cand1 (Wrong)
        [0.2, 0.8, 0.0], # Q1 -> Cand1 (Correct)
        [0.1, 0.2, 0.7]  # Q2 -> Cand2 (Correct)
    ])
    metrics = Metrics.recall_at_k(sim_matrix_bad, ground_truth, k_values=[1, 2])
    assert metrics['recall_at_1'] == 2/3
    # For Q0, correct answer (0) is rank 3 (score 0.1 vs 0.9, 0.0.. wait sorted: 0.9 (1), 0.1 (0), 0.0 (2). Rank 2)
    # Top 2 indices for Q0: [1, 0]. Correct is 0. So it is in top 2.
    assert metrics['recall_at_2'] == 1.0 

def test_vqa_accuracy():
    preds = ["blue", "red", "green"]
    truths = ["blue", "Red", "green."]
    # Normalization should handle case and punctuation
    assert Metrics.vqa_accuracy(preds, truths) == 1.0

