import numpy as np
from typing import List, Any, Dict
from sklearn.metrics import accuracy_score
import string

class Metrics:
    """
    Evaluation metrics for VLM tasks.
    """

    @staticmethod
    def accuracy(predictions: List[Any], references: List[Any]) -> float:
        return accuracy_score(references, predictions)

    @staticmethod
    def recall_at_k(similarity_matrix: np.ndarray, ground_truth_indices: List[int], k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Computes Recall@K for retrieval tasks.
        """
        n_queries = similarity_matrix.shape[0]
        results = {}
        
        max_k = max(k_values)
        
        # Sort predictions by score (descending)
        top_k_indices = np.argsort(-similarity_matrix, axis=1)[:, :max_k]
        
        for k in k_values:
            correct_count = 0
            for i in range(n_queries):
                if ground_truth_indices[i] in top_k_indices[i, :k]:
                    correct_count += 1
            results[f'recall_at_{k}'] = correct_count / n_queries
            
        return results

    @staticmethod
    def vqa_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
        """
        Computes VQA accuracy using exact match with normalization (lowercase, no punctuation).
        """
        def normalize_answer(s: str) -> str:
            def white_space_fix(text):
                return ' '.join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_punc(lower(s)))

        correct = 0
        for pred, truth in zip(predictions, ground_truths):
            if normalize_answer(pred) == normalize_answer(truth):
                correct += 1
        
        if not predictions:
            return 0.0
            
        return correct / len(predictions)
