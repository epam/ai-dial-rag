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
        KNOWN_MODELS, get_model_processor_classes, get_model_local_path
    )
    
    for model_name, model_type in KNOWN_MODELS.items():
        print(f"Downloading model {model_name} of type {model_type}")
        
        model_path = get_model_local_path(base_path, model_name)
        
        model_class, processor_class = get_model_processor_classes(model_type)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        # Download model and processor
        model = model_class.from_pretrained(model_name)
        processor = processor_class.from_pretrained(model_name)
        
        # Save model and processor
        model.save_pretrained(model_path)
        processor.save_pretrained(model_path)
        
        # saving just model and processor is not enough, we need to copy additional files
        print("Copying additional files...")
        important_files = [
            "preprocessor_config.json",
            "adapter_config.json",
            "processor_config.json",
        ]
        
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot_download(model_name, local_dir=temp_dir, local_files_only=False)
            for file_name in important_files:
                source_file = os.path.join(temp_dir, file_name)
                if os.path.exists(source_file):
                    shutil.copy2(source_file, f"{model_path}/{file_name}")
        
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
