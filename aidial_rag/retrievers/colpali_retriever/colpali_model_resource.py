# pyright: reportMissingImports=false

import threading

import torch

from aidial_rag.retrievers.colpali_retriever.colpali_index_config import (
    ColpaliIndexConfig,
    ColpaliModelType,
)


class ColpaliModelResource:
    def __init__(self, config: ColpaliIndexConfig | None):
        self.lock = threading.Lock()
        self.gpu_lock = threading.Lock()
        self.config: ColpaliIndexConfig | None = None
        self.model = None
        self.device: torch.device | None = None
        self.processor = None
        if config is not None:
            self.set_config(config)


    def get_gpu_lock(self):
        """Get the thread lock specifically for GPU operations."""
        return self.gpu_lock

    def set_config(self, config: ColpaliIndexConfig):
        config.validate_consistency()

        with self.lock:
            if self.config == config:
                return
            self.config = config
            device = "cpu"
            if torch.mps.is_available():
                device = "mps"
            if torch.cuda.is_available():
                device = "cuda"
            self.device = torch.device(device)

            from colpali_engine.models import (
                ColIdefics3,
                ColIdefics3Processor,
                ColPali,
                ColPaliProcessor,
                ColQwen2,
                ColQwen2Processor,
            )

            match config.model_type:
                case ColpaliModelType.COLPALI:
                    model_class = ColPali
                    processor_class = ColPaliProcessor
                case ColpaliModelType.COLIDEFICS:
                    model_class = ColIdefics3
                    processor_class = ColIdefics3Processor
                case ColpaliModelType.COLQWEN:
                    model_class = ColQwen2
                    processor_class = ColQwen2Processor
                case _:
                    raise ValueError("Invalid ColPali model type")
                
            self.model = model_class.from_pretrained(
                config.model_name, torch_dtype=torch.float16, device_map=device
            ).eval()
            self.processor = processor_class.from_pretrained(config.model_name)  # pyright: ignore

            assert self.model is not None
            assert self.processor is not None
            assert self.device is not None

    def get_model_processor_device(self):
        with self.lock:
            if self.config is None or self.device is None or self.model is None or self.processor is None:
                raise ValueError("ColpaliIndexConfig is required")
            return self.model, self.processor, self.device
