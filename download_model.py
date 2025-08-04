import sys
import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download
import tempfile
from sentence_transformers import SentenceTransformer


def download_model_for_backend(name: str, path: str, backend: str):
    print(f"Downloading model {name} to {path} with backend {backend}")
    model = SentenceTransformer(name, backend=backend)
    model.save(path)


def download_model(name: str, path: str, *backend_args: str):
    backends = backend_args or ["openvino"]
    print(f"Backends: {backends}")
    for backend in backends:
        download_model_for_backend(name, path, backend)


def download_all_colpali_models(base_path: str):
    """Download all models from the KNOWN_MODELS map"""
    print(f"Downloading all ColPali models to base path: {base_path}")
    
    # Import the KNOWN_MODELS map and utility functions from isolated module
    from aidial_rag.retrievers.colpali_retriever.colpali_models import (
        KNOWN_MODELS, get_model_processor_classes, get_model_local_path, get_model_cache_path
    )
    
    for model_name, model_type in KNOWN_MODELS.items(): 
        print(f"Downloading model {model_name} of type {model_type}")

        model_path = get_model_local_path(base_path, model_name)
        cache_path = get_model_cache_path(model_path)
        model_class, processor_class = get_model_processor_classes(model_type) 
        model_path.mkdir(parents=True, exist_ok=True)
        
        # download model repository for config files and adapters weights
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_name, local_dir=model_path, local_dir_use_symlinks=False, force_download=True)
        
        # download base model into cache directory
        model = model_class.from_pretrained(model_name,local_files_only=False,
            force_download=True,
            cache_dir=cache_path 
        )

        print(f"Successfully downloaded {model_name}")

def main():
    parser = argparse.ArgumentParser(
        description="Download models for AI-Dial-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all ColPali models
  python download_model.py colpali /path/to/colpali_models
  
  # Download embeddings model
  python download_model.py embeddings epam/bge-small-en /path/to/embeddings openvino torch
        """
    )
    
    # Create subparsers for different model types
    subparsers = parser.add_subparsers(dest='command', help='Model type to download')
    
    # ColPali models parser
    colpali_parser = subparsers.add_parser('colpali', help='Download all ColPali models')
    colpali_parser.add_argument('path', help='Base path to save all ColPali models')
    
    # Embeddings models parser  
    embeddings_parser = subparsers.add_parser('embeddings', help='Download embeddings model')
    embeddings_parser.add_argument('model_name', help='Hugging Face model name')
    embeddings_parser.add_argument('path', help='Path to save the model')
    embeddings_parser.add_argument('backends', nargs='*', default=['openvino'], 
                                 help='Backends to use (default: openvino)')
    
    args = parser.parse_args()
    
    if args.command == 'colpali':
        download_all_colpali_models(args.path)
    elif args.command == 'embeddings':
        download_model(args.model_name, args.path, *args.backends)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
