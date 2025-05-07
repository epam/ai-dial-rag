import sys

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


if __name__ == "__main__":
    download_model(*sys.argv[1:])
