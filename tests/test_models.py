import pytest
import torch
import numpy as np
from PIL import Image
from vlm_benchmark.models import CLIPModelWrapper

class MockCLIPModel:
    def __init__(self, **kwargs):
        self.logit_scale = torch.nn.Parameter(torch.tensor(np.log(100.0))) # scale exp() -> 100
    
    def to(self, device):
        # Mock the .to() method
        return self

    def get_image_features(self, **kwargs):
        # Return random features (batch, dim)
        # Access input size from kwargs to determine batch size
        # pixel_values is usually (batch, 3, h, w)
        batch_size = kwargs['pixel_values'].shape[0]
        return torch.randn(batch_size, 512)

    def get_text_features(self, **kwargs):
        batch_size = kwargs['input_ids'].shape[0]
        return torch.randn(batch_size, 512)
        
    def __call__(self, **kwargs):
        # This path is not used in batching optimization logic anymore but good for completeness
        pass

class MockProcessor:
    def __call__(self, text=None, images=None, return_tensors="pt", padding=True):
        class Batch(dict):
            # Need to inherit from dict for ** unpacking
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.__dict__ = self # allow attribute access
                
            def to(self, device):
                return self
        
        data = {}
        if text:
            data['input_ids'] = torch.zeros(len(text), 10)
            data['attention_mask'] = torch.ones(len(text), 10)
        if images:
            data['pixel_values'] = torch.zeros(len(images), 3, 224, 224)
            
        return Batch(**data)
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

@pytest.fixture
def mock_clip(monkeypatch):
    # Patch transformers loading so we don't need actual weights
    def mock_from_pretrained(*args, **kwargs):
        return MockCLIPModel()
    
    def mock_proc_from_pretrained(*args, **kwargs):
        return MockProcessor()

    monkeypatch.setattr("transformers.CLIPModel.from_pretrained", mock_from_pretrained)
    monkeypatch.setattr("transformers.CLIPProcessor.from_pretrained", mock_proc_from_pretrained)
    
    model = CLIPModelWrapper("dummy/clip", device="cpu")
    return model

def test_clip_wrapper_similarity(mock_clip):
    images = [Image.new('RGB', (224, 224)) for _ in range(4)]
    texts = ["a cat", "a dog", "a car"]
    
    # Should return numpy array (n_images, n_texts)
    sim_matrix = mock_clip.compute_image_text_similarity(images, texts)
    
    assert isinstance(sim_matrix, np.ndarray)
    assert sim_matrix.shape == (4, 3)
    
def test_clip_batching(mock_clip):
    # Force small batch size to test looping
    mock_clip.batch_size = 2
    
    images = [Image.new('RGB', (224, 224)) for _ in range(5)]
    texts = ["t1", "t2", "t3"]
    
    sim_matrix = mock_clip.compute_image_text_similarity(images, texts)
    assert sim_matrix.shape == (5, 3)
