import hashlib
import pickle
from pathlib import Path
from typing import Any, List, Tuple, Optional

import torch
from torch import Tensor

from aidial_rag.resources.colpali_model_resource import ColpaliModelResource
from aidial_rag.retrievers.colpali_retriever.colpali_index_config import ColpaliIndexConfig


class MockBatch:
    """Mock batch object that mimics PyTorch batch behavior."""
    
    def __init__(self, keys: List[str]):
        self.data = {}
        for key in keys:
            self.data[key] = self._create_tensor_for_key(key)
    
    def _create_tensor_for_key(self, key: str) -> Tensor:
        """Create appropriate tensor for the given key."""
        if key == 'input_ids':
            return torch.randint(0, 1000, (1, 10), dtype=torch.long)
        elif key == 'attention_mask':
            return torch.randint(0, 2, (1, 10), dtype=torch.long)
        elif key == 'pixel_values':
            return torch.randn(1, 3, 224, 224, dtype=torch.float32)
        else:
            return torch.randn(1, 10, dtype=torch.float32)
    
    def to(self, device) -> 'MockBatch':
        """Move all tensors to the specified device."""
        for key in self.data:
            self.data[key] = self.data[key].to(device)
        return self
    
    def keys(self):
        return self.data.keys()
    
    def __getitem__(self, key):
        return self.data[key]


class RecordingModel:
    """Wraps a real model to record all forward calls."""
    
    def __init__(self, real_model, cache_key: str, cache_dir: Path):
        self.real_model = real_model
        self.cache_key = cache_key
        self.cache_dir = cache_dir
        self.recorded_outputs = []
        self.call_count = 0
        
    def __call__(self, **kwargs):
        # Call the real model
        output = self.real_model(**kwargs)
        
        # Record the output
        self.recorded_outputs.append(output.detach().cpu())
        self.call_count += 1
        
        # Save to cache after each call
        self._save_recordings()
        
        return output
    
    def _save_recordings(self):
        """Save recorded outputs to cache."""
        cache_data = {
            'model_outputs': self.recorded_outputs,
            'call_count': self.call_count
        }
        cache_path = self.cache_dir / f"{self.cache_key}_model.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)


class RecordingScoreProcessor:
    """Wraps a real processor to record only score calls."""
    
    def __init__(self, real_processor, cache_key: str, cache_dir: Path):
        self.real_processor = real_processor
        self.cache_key = cache_key
        self.cache_dir = cache_dir
        self.recorded_scores = []
        
    def process_queries(self, queries: List[str]):
        """Use real processor for queries processing."""
        return self.real_processor.process_queries(queries)
    
    def process_images(self, images: List[Any]):
        """Use real processor for images processing."""
        return self.real_processor.process_images(images)
    
    def score(self, query_embeddings: Tensor, image_embeddings: List[Tensor]):
        """Call real processor for scoring and record the result."""
        output = self.real_processor.score(query_embeddings, image_embeddings)
        self._record_score(output)
        return output
    
    def score_multi_vector(self, query_embeddings: Tensor, image_embeddings: List[Tensor]):
        """Call real processor for multi-vector scoring and record the result."""
        output = self.real_processor.score_multi_vector(query_embeddings, image_embeddings)
        self._record_score(output)
        return output
    
    def _record_score(self, score: Tensor):
        """Record a score and save to cache."""
        self.recorded_scores.append(score)
        self._save_recordings()
    
    def _save_recordings(self):
        """Save recorded scores to cache."""
        cache_data = {'scores': self.recorded_scores}
        cache_path = self.cache_dir / f"{self.cache_key}_scores.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(cache_data, f)


class ReplayModel:
    """Replays recorded model outputs."""
    
    def __init__(self, recorded_outputs: List[Tensor]):
        self.recorded_outputs = recorded_outputs
        self.call_count = 0
        
    def __call__(self, **kwargs):
        if self.call_count < len(self.recorded_outputs):
            output = self.recorded_outputs[self.call_count]
            self.call_count += 1
            return output
        else:
            # If we run out of recordings, cycle back
            self.call_count = 0
            return self.recorded_outputs[0] if self.recorded_outputs else torch.zeros(1, 768)


class ReplayScoreProcessor:
    """Replays recorded scores with mock processing."""
    
    def __init__(self, recorded_scores: List[Tensor]):
        self.recorded_scores = recorded_scores
        self.score_count = 0
        
    def process_queries(self, queries: List[str]):
        """Return mock batch for queries processing."""
        return MockBatch(['input_ids', 'attention_mask'])
    
    def process_images(self, images: List[Any]):
        """Return mock batch for images processing."""
        return MockBatch(['input_ids', 'attention_mask', 'pixel_values'])
    
    def score(self, query_embeddings: Tensor, image_embeddings: List[Tensor]):
        """Replay recorded scores."""
        return self._get_next_score()
    
    def score_multi_vector(self, query_embeddings: Tensor, image_embeddings: List[Tensor]):
        """Replay recorded scores for multi-vector scoring."""
        return self._get_next_score()
    
    def _get_next_score(self) -> Tensor:
        """Get the next recorded score, cycling back if needed."""
        if self.score_count < len(self.recorded_scores):
            output = self.recorded_scores[self.score_count]
            self.score_count += 1
            return output
        else:
            # If we run out of recordings, cycle back
            self.score_count = 0
            return self.recorded_scores[0] if self.recorded_scores else torch.rand(1, 1)


class CachedColpaliModelResource(ColpaliModelResource):
    """
    A cached version of ColpaliModelResource that records real model outputs and scores
    when use_cache=False and replays them when use_cache=True.
    """
    
    def __init__(self, use_cache: bool = True, cache_dir: str = "tests/cache/test_colpali_retriever"):
        super().__init__()
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Recording/replay objects
        self._recording_model: Optional[RecordingModel] = None
        self._recording_processor: Optional[RecordingScoreProcessor] = None
        self._replay_model: Optional[ReplayModel] = None
        self._replay_processor: Optional[ReplayScoreProcessor] = None
    
    def _get_cache_key(self, model_name: str, model_type: str) -> str:
        """Generate cache key for model configuration."""
        content = f"{model_name}_{model_type}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _load_recordings(self, cache_key: str) -> Tuple[Optional[List[Tensor]], Optional[List[Tensor]]]:
        """Load recorded model outputs and scores."""
        model_cache_path = self.cache_dir / f"{cache_key}_model.pkl"
        scores_cache_path = self.cache_dir / f"{cache_key}_scores.pkl"
        
        model_outputs = None
        recorded_scores = None
        
        if model_cache_path.exists():
            with open(model_cache_path, 'rb') as f:
                model_data = pickle.load(f)
                model_outputs = model_data.get('model_outputs', [])
        
        if scores_cache_path.exists():
            with open(scores_cache_path, 'rb') as f:
                scores_data = pickle.load(f)
                recorded_scores = scores_data.get('scores', [])
        
        return model_outputs, recorded_scores
    
    def get_model_processor_device(self, config: ColpaliIndexConfig) -> tuple[Any, Any, torch.device]:
        """Get model, processor, and device with recording/replay support."""
        cache_key = self._get_cache_key(config.model_name, str(config.model_type))
        
        if self.use_cache:
            return self._get_replay_objects(config, cache_key)
        else:
            return self._get_recording_objects(config, cache_key)
    
    def _get_replay_objects(self, config: ColpaliIndexConfig, cache_key: str) -> tuple[Any, Any, torch.device]:
        """Get replay objects for cached mode."""
        model_outputs, recorded_scores = self._load_recordings(cache_key)
        
        if model_outputs is not None and recorded_scores is not None:
            print(f"Replaying recorded outputs for {config.model_name}")
            self._replay_model = ReplayModel(model_outputs)
            self._replay_processor = ReplayScoreProcessor(recorded_scores)
            device = torch.device('cpu')  # Default for replay
            return self._replay_model, self._replay_processor, device
        else:
            print(f"No recordings found for {config.model_name}, falling back to real model")
            return super().get_model_processor_device(config)
    
    def _get_recording_objects(self, config: ColpaliIndexConfig, cache_key: str) -> tuple[Any, Any, torch.device]:
        """Get recording objects for recording mode."""
        print(f"Recording real model outputs for {config.model_name}")
        real_model, real_processor, device = super().get_model_processor_device(config)
        
        # Wrap with recording objects
        self._recording_model = RecordingModel(real_model, cache_key, self.cache_dir)
        self._recording_processor = RecordingScoreProcessor(real_processor, cache_key, self.cache_dir)
        
        return self._recording_model, self._recording_processor, device
    
    def get_recorded_outputs(self) -> List[Tensor]:
        """Get the recorded model outputs for testing."""
        if self._recording_model:
            return self._recording_model.recorded_outputs
        elif self._replay_model:
            return self._replay_model.recorded_outputs
        return []
    
    def get_recorded_scores(self) -> List[Tensor]:
        """Get the recorded scores for testing."""
        if self._recording_processor:
            return self._recording_processor.recorded_scores
        elif self._replay_processor:
            return self._replay_processor.recorded_scores
        return [] 