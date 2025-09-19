import sys
import argparse
from huggingface_hub import snapshot_download


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
    model_class.from_pretrained(model_name,local_files_only=False,
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
    # Download a specific ColPali model
    python download_colpali_models.py vidore/colpali-v1.3 /path/to/colpali_models
        """
    )
    
    parser.add_argument('model_name', help='ColPali model name to download')
    parser.add_argument('path', help='Base path to save ColPali model')
    
    args = parser.parse_args()
    download_colpali_model(args.path, args.model_name)


if __name__ == "__main__":
    main()
