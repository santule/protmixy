"""
Model loader module for MSA-Transformer.

This module implements a singleton pattern to load the MSA-Transformer model
only once and reuse it across multiple function calls, improving efficiency.
"""
import torch
from esm import pretrained
from config.settings import DEVICE, GENERATOR_MODEL_NAME


class ModelLoader:
    """
    Singleton class for loading and caching the MSA-Transformer model.
    
    The model is loaded once on first access and cached for subsequent uses,
    avoiding redundant loading operations.
    
    Attributes:
        _model: Cached MSA-Transformer model instance
        _alphabet: Cached alphabet for tokenization
    """
    
    _model = None
    _alphabet = None

    @staticmethod
    def get_model():
        """
        Get the MSA-Transformer model and alphabet.
        
        Loads the model on first call and returns cached version on subsequent calls.
        The model is set to evaluation mode and moved to the configured device.
        
        Returns:
            tuple: (model, alphabet) where:
                - model: MSA-Transformer model instance
                - alphabet: Tokenization alphabet
        """
        if ModelLoader._model is None or ModelLoader._alphabet is None:
            print(f"Loading MSA-Transformer model: {GENERATOR_MODEL_NAME}...")
            ModelLoader._model, ModelLoader._alphabet = pretrained.load_model_and_alphabet(
                GENERATOR_MODEL_NAME
            )
            ModelLoader._model.eval()
            ModelLoader._model.to(DEVICE)
            print(f"Model loaded successfully on device: {DEVICE}")
        
        return ModelLoader._model, ModelLoader._alphabet
