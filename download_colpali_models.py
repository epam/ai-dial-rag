import sys
import argparse
import yaml
from pathlib import Path
from huggingface_hub import snapshot_download


def download_colpali_model_from_config(config_path: str):
    """Download a ColPali model using configuration from a YAML file"""
    # Load the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract model information from config
    model_resource_config = config.get('colpali_model_resource_config', {})
    model_name = model_resource_config.get('model_name')
    base_path = model_resource_config.get('models_folder_path')
    
    if not model_name:
        raise ValueError("model_name not found in colpali_model_resource_config")
    if not base_path:
        raise ValueError("models_folder_path not found in colpali_model_resource_config")
    
    return download_colpali_model(base_path, model_name)


def download_colpali_model(base_path: str, model_name: str):
    """Download a single ColPali model from the KNOWN_MODELS map"""
    # Import the KNOWN_MODELS map and utility functions
    from aidial_rag.retrievers.colpali_retriever.colpali_models import (
        MODEL_NAME_TO_TYPE, get_model_processor_classes, get_model_local_path, get_model_cache_path
    )
    
    # Validate model name
    if model_name not in MODEL_NAME_TO_TYPE:
        raise ValueError(f"Model '{model_name}' not found in known models: {list(MODEL_NAME_TO_TYPE.keys())}")
    
    print(f"Downloading ColPali model '{model_name}' to base path: {base_path}")
    
    model_path = get_model_local_path(base_path, model_name)
    cache_path = get_model_cache_path(model_path)
    model_class, _ = get_model_processor_classes(model_name) 
    model_path.mkdir(parents=True, exist_ok=True)
    
    # download model repository for config files and adapters weights
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False, force_download=True)
    
    # download base model into cache directory
    model_class.from_pretrained(model_name,
        local_files_only=False,
        force_download=True,
        cache_dir=cache_path
    )

    print(f"Successfully downloaded {model_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Download ColPali models for AI-Dial-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download using config file
    python download_colpali_models.py config/azure_colsmol256m.yaml
        """
    )
    
    parser.add_argument('config', help='Path to YAML config file')
    
    args = parser.parse_args()
    download_colpali_model_from_config(args.config)


if __name__ == "__main__":
    main()
