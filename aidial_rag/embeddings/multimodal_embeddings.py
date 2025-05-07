import numpy as np
from langchain_openai import AzureOpenAIEmbeddings
from openai.resources import AsyncEmbeddings

from aidial_rag.dial_config import DialConfig

MULTIMODAL_EMBEDDING_TIMEOUT: float = 60


class MultimodalEmbeddings(AzureOpenAIEmbeddings):
    def __init__(
        self,
        dial_config: DialConfig,
        embeddings_model_name: str,
        max_retries: int = 0,
    ):
        super().__init__(
            azure_endpoint=dial_config.dial_url,
            api_key=dial_config.api_key,
            api_version="2023-03-15-preview",
            azure_deployment=embeddings_model_name,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
            max_retries=max_retries,
        )

    async def aembed_image(self, image: str) -> np.ndarray:
        if self.deployment is None:
            raise ValueError("Deployment name cannot be None")
        async_embeddings: AsyncEmbeddings = self.async_client
        response = await async_embeddings.create(
            model=self.deployment,
            input=[],
            extra_body={
                "custom_input": [
                    {
                        "type": "image/png",
                        "data": image,
                    }
                ],
            },
            encoding_format="float",
            timeout=MULTIMODAL_EMBEDDING_TIMEOUT,
        )
        assert len(response.data) == 1
        return np.array(response.data[0].embedding)
