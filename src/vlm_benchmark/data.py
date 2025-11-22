from datasets import load_dataset as hf_load_dataset, Dataset, Image, DatasetDict
from typing import Optional, Tuple, Dict, Any
import os

def load_dataset(file_path: str) -> Dataset:
    """
    Loads a dataset from a JSONL file.
    """
    print(f"Loading dataset from: {file_path}")
    
    dataset = hf_load_dataset("json", data_files=file_path, split="train")
    
    base_dir = os.path.dirname(file_path)
    
    def resolve_image_path(example: Dict[str, Any]) -> Dict[str, Any]:
        if not os.path.isabs(example['image_path']):
            example['image_path'] = os.path.join(base_dir, example['image_path'])
        return example

    dataset = dataset.map(resolve_image_path)
    dataset = dataset.cast_column("image_path", Image())
    
    return dataset

def load_example_dataset() -> Dataset:
    """
    Loads the included example dataset.
    """
    example_path = os.path.join("data", "example", "annotations.jsonl")
    if not os.path.exists(example_path):
        raise FileNotFoundError(f"Example dataset not found at {example_path}")
    
    return load_dataset(example_path)

def make_splits(dataset: Dataset, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> DatasetDict:
    """
    Splits a Dataset into train, validation, and test sets.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")

    test_val_ratio = val_ratio + test_ratio
    if test_val_ratio == 0:
         return DatasetDict({'train': dataset})
         
    train_test_split = dataset.train_test_split(test_size=test_val_ratio, seed=seed)
    train_dataset = train_test_split['train']
    rest_dataset = train_test_split['test']
    
    if test_ratio == 0:
        return DatasetDict({
            'train': train_dataset,
            'validation': rest_dataset
        })
        
    relative_test_ratio = test_ratio / test_val_ratio
    val_test_split = rest_dataset.train_test_split(test_size=relative_test_ratio, seed=seed)
    
    return DatasetDict({
        'train': train_dataset,
        'validation': val_test_split['train'],
        'test': val_test_split['test']
    })

class DatasetLoader:
    """
    Loader for datasets (local or Hugging Face Hub).
    """
    def __init__(self, dataset_name: str, split: str = "test"):
        self.dataset_name = dataset_name
        self.split = split

    def load(self, subset: Optional[str] = None) -> Optional[Dataset]:
        print(f"Loading dataset: {self.dataset_name} (split={self.split})")
        try:
            if self.dataset_name.endswith(".jsonl") and os.path.exists(self.dataset_name):
                 return load_dataset(self.dataset_name)
            
            if subset:
                dataset = hf_load_dataset(self.dataset_name, subset, split=self.split)
            else:
                dataset = hf_load_dataset(self.dataset_name, split=self.split)
            return dataset
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
