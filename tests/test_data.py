import pytest
import os
import json
from PIL import Image
from vlm_benchmark.data import load_dataset, make_splits, DatasetLoader

@pytest.fixture
def dummy_jsonl(tmp_path):
    # Create dummy data
    base_dir = tmp_path / "data"
    base_dir.mkdir()
    
    image_files = []
    data = []
    for i in range(10):
        img_name = f"img_{i}.jpg"
        img_path = base_dir / img_name
        Image.new('RGB', (10, 10), color='red').save(img_path)
        
        data.append({
            "image_path": img_name,
            "text": f"text {i}",
            "label": i
        })
        
    jsonl_path = base_dir / "data.jsonl"
    with open(jsonl_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
            
    return str(jsonl_path)

def test_load_dataset(dummy_jsonl):
    dataset = load_dataset(dummy_jsonl)
    assert len(dataset) == 10
    # Check if image is loaded as PIL Image
    assert isinstance(dataset[0]['image_path'], Image.Image)
    assert dataset[0]['text'] == "text 0"

def test_make_splits(dummy_jsonl):
    dataset = load_dataset(dummy_jsonl)
    splits = make_splits(dataset, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
    
    assert 'train' in splits
    assert 'validation' in splits
    assert 'test' in splits
    
    assert len(splits['train']) == 6
    assert len(splits['validation']) == 2
    assert len(splits['test']) == 2

def test_dataset_loader_local(dummy_jsonl):
    loader = DatasetLoader(dummy_jsonl)
    dataset = loader.load()
    assert len(dataset) == 10

