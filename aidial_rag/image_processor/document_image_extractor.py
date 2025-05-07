from abc import ABC, abstractmethod
from typing import AsyncGenerator, Iterable, List, Optional

from PIL.Image import Image

from aidial_rag.utils import check_mime_type


class DocumentPageImageExtractor(ABC):
    supported_mime_types: List[str]

    def get_supported_mime_types(self):
        return self.supported_mime_types

    def is_mime_supported(self, mime: str):
        return check_mime_type(mime, self.supported_mime_types)

    @abstractmethod
    def get_number_of_pages(self, file_bytes: bytes) -> int:
        """
        Get number of pages for given document
        Parameters:
            file_bytes (bytes): The file content as bytes.
        Returns:
            int: Count of pages in the document.
        """
        pass

    @abstractmethod
    def extract_pages_gen(
        self,
        file_bytes: bytes,
        page_numbers: Iterable[int],
        scaled_size: Optional[int] = None,
    ) -> AsyncGenerator[Image, None]:
        """
        Extracts specified pages image from a document.

        Parameters:
            file_bytes (bytes): The file content as bytes.
            page_numbers (Iterable[int]): Page ordinal numbers array (1 for first page)
            scaled_size (Optional[int]): Size to scale the image (if provided).

        Returns:
            AsyncGenerator[Image, None]: A generator to iterate over the extracted of images.

        Raises:
            RuntimeError: If an invalid page number is specified.
        """
        pass
