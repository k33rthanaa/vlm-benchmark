from abc import ABC, abstractmethod
from typing import Dict, Any, List
from .models import VLM, CLIPModelWrapper, VQAModelWrapper
from .data import DatasetLoader
from .metrics import Metrics
import numpy as np

class Task(ABC):
    """
    Abstract Base Class for evaluation tasks.
    """
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run(self, model: VLM, dataset_loader: DatasetLoader) -> Dict[str, Any]:
        pass

def run_image_text_retrieval(clip_model: CLIPModelWrapper, dataset: Any, k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Runs image-text retrieval evaluation.
    Computes Recall@K metrics.
    """
    print("Preparing data for retrieval...")
    
    # Identify correct feature keys
    image_key = 'image' if 'image' in dataset.features else 'image_path'
    text_key = 'text' if 'text' in dataset.features else 'caption'
    
    images = dataset[image_key]
    texts = dataset[text_key]
    
    print(f"Computing similarity for {len(images)} images and {len(texts)} texts...")
    similarity_matrix = clip_model.compute_image_text_similarity(images, texts)
    
    # Assumption: 1-to-1 mapping between images and texts
    ground_truth_indices = list(range(len(images)))
    
    metrics = Metrics.recall_at_k(similarity_matrix, ground_truth_indices, k_values)
    return metrics

def run_vqa(vqa_model: VQAModelWrapper, dataset: Any) -> Dict[str, float]:
    """
    Runs Visual Question Answering evaluation.
    """
    print("Running VQA evaluation...")
    
    image_key = 'image' if 'image' in dataset.features else 'image_path'
    question_key = 'question' if 'question' in dataset.features else 'text'
    answer_key = 'answer' if 'answer' in dataset.features else 'label'
    
    images = dataset[image_key]
    questions = dataset[question_key]
    ground_truths = dataset[answer_key]
    
    predictions = vqa_model.answer_questions(images, questions)
    
    # Handle various ground truth formats (string or list of strings)
    normalized_truths = []
    for gt in ground_truths:
        if isinstance(gt, list):
             if len(gt) > 0:
                 normalized_truths.append(str(gt[0]))
             else:
                 normalized_truths.append("")
        else:
            normalized_truths.append(str(gt))
            
    accuracy = Metrics.vqa_accuracy(predictions, normalized_truths)
    
    return {"accuracy": accuracy}

class ImageTextRetrievalTask(Task):
    def __init__(self):
        super().__init__("image_text_retrieval")

    def run(self, model: VLM, dataset_loader: DatasetLoader) -> Dict[str, Any]:
        dataset = dataset_loader.load()
        if not dataset:
            return {"error": "Dataset not loaded"}

        if isinstance(model, CLIPModelWrapper):
            return run_image_text_retrieval(model, dataset)
        else:
            raise ValueError("ImageTextRetrievalTask requires a CLIPModelWrapper")

class VQATask(Task):
    def __init__(self):
        super().__init__("vqa")
        
    def run(self, model: VLM, dataset_loader: DatasetLoader) -> Dict[str, Any]:
        dataset = dataset_loader.load()
        if not dataset:
            return {"error": "Dataset not loaded"}
            
        if isinstance(model, VQAModelWrapper):
            return run_vqa(model, dataset)
        else:
             raise ValueError("VQATask requires a VQAModelWrapper")

class TaskFactory:
    @staticmethod
    def create_task(task_name: str) -> Task:
        if task_name == "retrieval":
            return ImageTextRetrievalTask()
        elif task_name == "vqa":
            return VQATask()
        else:
            raise ValueError(f"Unknown task: {task_name}")
