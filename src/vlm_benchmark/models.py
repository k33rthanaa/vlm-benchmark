from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any
from PIL import Image
import torch
import numpy as np
from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotImageClassification, 
    AutoModelForVisualQuestionAnswering, 
    CLIPModel, 
    CLIPProcessor,
    ViltProcessor, 
    ViltForQuestionAnswering,
    BlipProcessor, 
    BlipForQuestionAnswering
)

class VLM(ABC):
    """
    Abstract Base Class for Vision-Language Models.
    """
    @abstractmethod
    def load_model(self) -> None:
        """
        Loads the model and processor/tokenizer into memory and moves them to the appropriate device.
        """
        pass

    @abstractmethod
    def predict(self, images: List[Image.Image], texts: List[str], task: str) -> Any:
        """
        Performs prediction on a batch of images and texts for a specific task.

        Args:
            images (List[Image.Image]): A list of PIL Images.
            texts (List[str]): A list of text inputs (captions, questions, etc.).
            task (str): The task identifier (e.g., 'retrieval', 'vqa').

        Returns:
            Any: The prediction results, format depends on the task.
        """
        pass

class CLIPModelWrapper(VLM):
    """
    Wrapper for CLIP-like models (Contrastive Language-Image Pre-training).
    Primarily used for image-text retrieval and zero-shot classification.
    """
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[str] = None, batch_size: int = 32):
        """
        Initializes the CLIP wrapper.

        Args:
            model_name (str): The Hugging Face model ID. Defaults to "openai/clip-vit-base-patch32".
            device (Optional[str]): The device to run on ('cpu' or 'cuda'). If None, automatically detects.
            batch_size (int): Batch size for processing embeddings. Defaults to 32.
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """Loads the CLIP model and processor."""
        print(f"Loading CLIP model: {self.model_name}")
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def predict(self, images: List[Image.Image], texts: List[str], task: str = "retrieval") -> Any:
        """
        Routes predictions based on the task.
        """
        if task == "retrieval":
            return self.compute_image_text_similarity(images, texts)
        else:
            raise NotImplementedError(f"Task {task} not supported by CLIPModelWrapper.")

    def compute_image_text_similarity(self, images: List[Image.Image], texts: List[str]) -> np.ndarray:
        """
        Computes the similarity matrix between a list of images and a list of texts.
        Handles batch processing to avoid OOM on large inputs.

        Args:
            images (List[Image.Image]): List of PIL images.
            texts (List[str]): List of text strings.

        Returns:
            np.ndarray: A similarity matrix of shape (n_images, n_texts). 
                        Values are logits (dot product of normalized embeddings * logit_scale).
        """
        if self.model is None:
            self.load_model()
            
        n_images = len(images)
        n_texts = len(texts)
        
        # Pre-compute text embeddings
        text_embeddings = []
        for i in range(0, n_texts, self.batch_size):
            batch_texts = texts[i:i + self.batch_size]
            text_inputs = self.processor(text=batch_texts, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                batch_text_emb = self.model.get_text_features(**text_inputs)
                batch_text_emb = batch_text_emb / batch_text_emb.norm(p=2, dim=-1, keepdim=True)
                text_embeddings.append(batch_text_emb)
        
        if text_embeddings:
            all_text_embeddings = torch.cat(text_embeddings, dim=0)
        else:
            return np.array([[]])

        # Compute image embeddings
        image_embeddings = []
        for i in range(0, n_images, self.batch_size):
            batch_images = images[i:i + self.batch_size]
            image_inputs = self.processor(images=batch_images, return_tensors="pt").to(self.device)
            with torch.no_grad():
                batch_image_emb = self.model.get_image_features(**image_inputs)
                batch_image_emb = batch_image_emb / batch_image_emb.norm(p=2, dim=-1, keepdim=True)
                image_embeddings.append(batch_image_emb)
        
        if image_embeddings:
             all_image_embeddings = torch.cat(image_embeddings, dim=0)
        else:
             return np.array([[]])

        # Compute similarity matrix (scaled dot product)
        logit_scale = self.model.logit_scale.exp()
        logits_per_image = logit_scale * torch.matmul(all_image_embeddings, all_text_embeddings.t())
        
        return logits_per_image.detach().cpu().numpy()

class VQAModelWrapper(VLM):
    """
    Wrapper for Visual Question Answering (VQA) models (e.g., ViLT, BLIP).
    """
    def __init__(self, model_name: str = "dandelin/vilt-b32-finetuned-vqa", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None

    def load_model(self) -> None:
        """
        Loads the VQA model and processor. 
        """
        print(f"Loading VQA model: {self.model_name}")
        
        if "vilt" in self.model_name.lower():
            self.processor = ViltProcessor.from_pretrained(self.model_name)
            self.model = ViltForQuestionAnswering.from_pretrained(self.model_name).to(self.device)
        elif "blip" in self.model_name.lower():
            self.processor = BlipProcessor.from_pretrained(self.model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(self.model_name).to(self.device)
        else:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForVisualQuestionAnswering.from_pretrained(self.model_name).to(self.device)

    def predict(self, images: List[Image.Image], texts: List[str], task: str = "vqa") -> List[str]:
        if task == "vqa":
            return self.answer_questions(images, texts)
        else:
            raise NotImplementedError(f"Task {task} not supported by VQAModelWrapper.")

    def answer_questions(self, images: List[Image.Image], questions: List[str]) -> List[str]:
        """
        Generates answers for the given image-question pairs.
        """
        if self.model is None:
            self.load_model()
            
        # Broadcast if necessary (e.g. 1 question for N images)
        if len(images) != len(questions):
            if len(questions) == 1:
                 questions = questions * len(images)
            elif len(images) == 1:
                 images = images * len(questions)
            else:
                raise ValueError("Length of images and questions must match or be broadcastable.")

        answers = []
        
        # Iterative inference
        for img, q in zip(images, questions):
            inputs = self.processor(images=img, text=q, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            
            if hasattr(self.model.config, "id2label"):
                 answer = self.model.config.id2label[idx]
            else:
                 answer = str(idx)
                 
            answers.append(answer)
            
        return answers

class ModelFactory:
    """
    Factory for creating VLM instances.
    """
    @staticmethod
    def create_model(model_type: str, model_name: str) -> VLM:
        if model_type == "clip":
            return CLIPModelWrapper(model_name)
        elif model_type == "vqa":
            return VQAModelWrapper(model_name)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
