import argparse
import yaml
from aidial_rag.retrievers.colpali_retriever.colpali_models import (
    load_model_and_processor
)

import torch

def download_colpali_model_from_config(config_path: str):
    """Download a ColPali model using configuration from a YAML file"""
    # Load the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract model information from config
    model_resource_config = config.get('colpali_model_resource_config', {})
    model_name = model_resource_config.get('model_name')

    if not model_name:
        raise ValueError("model_name not found in colpali_model_resource_config")

    device = torch.device("cpu")
    # after loading model it will cache it and next calls will use cached model
    load_model_and_processor(model_name, device)
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
