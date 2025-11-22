import os
import json
import time
import torch
import random
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional
from .models import ModelFactory
from .data import DatasetLoader, make_splits
from .tasks import TaskFactory

@dataclass
class BenchmarkConfig:
    """
    Configuration for running a VLM benchmark.
    """
    dataset_name: str
    tasks: List[str]
    model_type: str  # General type like 'clip' or 'vqa'
    model_name: str  # Specific Hugging Face model name
    seed: int = 42
    output_dir: str = "results"
    split_ratios: Dict[str, float] = field(default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1})

def set_seed(seed: int) -> None:
    """
    Sets random seeds for Python, NumPy, and PyTorch to ensure reproducibility.
    
    Args:
        seed (int): The seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """
    Runs the benchmark suite based on the provided configuration.

    This function orchestrates the entire benchmark process:
    1. Sets random seeds.
    2. Loads and prepares the dataset (including splitting if necessary).
    3. Instantiates and loads the specified model.
    4. Runs each configured task.
    5. Aggregates and saves results to a JSON file.

    Args:
        config (BenchmarkConfig): The configuration object.

    Returns:
        Dict[str, Any]: A dictionary containing the configuration and results for all tasks.
    """
    # 1. Set deterministic seed
    set_seed(config.seed)
    print(f"Starting benchmark with seed {config.seed}")
    print(f"Configuration: {config}")

    # 2. Load Dataset
    # Use DatasetLoader wrapper but also leverage make_splits if needed
    loader = DatasetLoader(config.dataset_name)
    # We load the 'test' split by default in current loader logic, or 'train' if local file.
    # Let's load the base dataset first.
    
    # For local files, DatasetLoader returns a dataset with 'train' split usually
    # If it's HF dataset, it respects split arg.
    # To generalize, let's assume we want to run on the 'test' equivalent.
    
    if config.dataset_name.endswith('.jsonl'):
        # It's a local file, load and split
        full_dataset = loader.load()
        if full_dataset is None:
             print("Failed to load dataset.")
             return {}

        # Create splits if they don't exist in the object (HF dataset object from json is single split)
        # The make_splits function returns a DatasetDict
        dataset_splits = make_splits(
            full_dataset, 
            train_ratio=config.split_ratios.get('train', 0.8),
            val_ratio=config.split_ratios.get('val', 0.1),
            test_ratio=config.split_ratios.get('test', 0.1),
            seed=config.seed
        )
        test_dataset = dataset_splits['test']
        print(f"Created splits. Using test split with {len(test_dataset)} examples.")
    else:
        # HF Dataset, just load test split
        test_dataset = loader.load(subset=None) # subset logic is inside loader if needed, here simplified
        if test_dataset is None:
             print("Failed to load dataset.")
             return {}
        print(f"Loaded existing test split with {len(test_dataset)} examples.")

    # 3. Instantiate Model
    try:
        model = ModelFactory.create_model(config.model_type, config.model_name)
        model.load_model()
    except Exception as e:
        print(f"Error loading model: {e}")
        return {"error": str(e)}

    # 4. Run Tasks
    results = {}
    results['config'] = asdict(config)
    results['metrics'] = {}

    # Adapter to reuse the Task.run interface which expects a DatasetLoader
    class TestSplitLoader:
        def load(self):
            return test_dataset
    
    test_loader_adapter = TestSplitLoader()

    for task_name in config.tasks:
        print(f"Running task: {task_name}")
        try:
            task = TaskFactory.create_task(task_name)
            task_metrics = task.run(model, test_loader_adapter)
            results['metrics'][task_name] = task_metrics
            print(f"Metrics for {task_name}: {task_metrics}")
        except Exception as e:
            print(f"Error running task {task_name}: {e}")
            results['metrics'][task_name] = {"error": str(e)}

    # 5. Save Results
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = int(time.time())
    output_file = os.path.join(config.output_dir, f"{timestamp}.json")
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results

class Evaluator:
    """
    Orchestrator for evaluating a single model on a single task and dataset.
    Currently used primarily by legacy CLI arguments.
    """
    def __init__(self, model_type: str, model_name: str, task_name: str, dataset_name: str):
        """
        Initializes the Evaluator.

        Args:
            model_type (str): Model type ('clip', 'vqa').
            model_name (str): Hugging Face model name.
            task_name (str): Task name ('retrieval', 'vqa').
            dataset_name (str): Dataset path or name.
        """
        self.model = ModelFactory.create_model(model_type, model_name)
        self.task = TaskFactory.create_task(task_name)
        self.dataset_loader = DatasetLoader(dataset_name)

    def evaluate(self) -> Dict[str, Any]:
        """
        Runs the evaluation.

        Returns:
            Dict[str, Any]: Evaluation results.
        """
        self.model.load_model()
        results = self.task.run(self.model, self.dataset_loader)
        return results
