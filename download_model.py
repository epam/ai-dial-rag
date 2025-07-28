import sys
import argparse
from pathlib import Path

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
    
    # Import the KNOWN_MODELS map and utility functions
    from aidial_rag.retrievers.colpali_retriever.colpali_model_resource import (
        KNOWN_MODELS, get_model_processor_classes, get_model_local_path
    )
    
    for model_name, model_type in KNOWN_MODELS.items():
        model_path = get_model_local_path(base_path, model_name)

        model_class, processor_class = get_model_processor_classes(model_type)
        Path(model_path).mkdir(parents=True, exist_ok=True)
        
        model = model_class.from_pretrained(model_name)
        processor = processor_class.from_pretrained(model_name)
        
        model.save_pretrained(model_path)
        processor.save_pretrained(model_path)


def main():
    parser = argparse.ArgumentParser(
        description="Download models for AI-Dial-RAG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all ColPali models
  python download_model.py --colpali /path/to/colpali_models
  
  # Download embeddings model
  python download_model.py --embeddings epam/bge-small-en /path/to/embeddings openvino torch
        """
    )
    
    # Create subparsers for different model types
    subparsers = parser.add_subparsers(dest='command', help='Model type to download')
    
    # ColPali models parser
    colpali_parser = subparsers.add_parser('--colpali', help='Download all ColPali models')
    colpali_parser.add_argument('path', help='Base path to save all ColPali models')
    
    # Embeddings models parser  
    embeddings_parser = subparsers.add_parser('--embeddings', help='Download embeddings model')
    embeddings_parser.add_argument('model_name', help='Hugging Face model name')
    embeddings_parser.add_argument('path', help='Path to save the model')
    embeddings_parser.add_argument('backends', nargs='*', default=['openvino'], 
                                 help='Backends to use (default: openvino)')
    
    args = parser.parse_args()
    
    if args.command == '--colpali':
        download_all_colpali_models(args.path)
    elif args.command == '--embeddings':
        download_model(args.model_name, args.path, *args.backends)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
