import base64
import json

from aidial_sdk.utils.merge_chunks import merge_chat_completion_chunks
from httpx import Response


def filter_out(headers):
    filtered_headers = [
        "content-length",
    ]
    return {
        key: value
        for key, value in headers.items()
        if key not in filtered_headers
    }


class CacheResponse:
    status_code: int
    headers: dict
    body: str = ""
    body_encoded: str = ""
    request_query: str
    request_body: str

    @classmethod
    def from_response(
        cls,
        response: Response,
        body: bytes,
        request_query: str,
        request_body: str,
    ):
        body_encoded = ""
        try:
            body_str = body.decode()
        except UnicodeDecodeError:
            body_encoded = base64.b64encode(body).decode("utf-8")
            body_str = ""
        return cls(
            status_code=response.status_code,
            headers=filter_out(response.headers),
            body=body_str,
            body_encoded=body_encoded,
            request_query=request_query,
            request_body=request_body,
        )

    def __init__(
        self,
        status_code: int,
        headers: dict,
        body: str,
        body_encoded: str,
        request_query: str,
        request_body: str,
    ):
        self.status_code = status_code
        self.headers = headers
        self.body = body
        self.body_encoded = body_encoded
        self.request_query = request_query
        self.request_body = request_body
        super().__init__()

    def serialize(self) -> str:
        """Serialize the instance to a JSON string."""
        return json.dumps(
            {
                "body_human_readable": self.body_human_readable(),
                "request": {
                    "query": self.request_query,
                    "body": json.loads(self.request_body),
                },
                "response": {
                    "status_code": self.status_code,
                    "headers": filter_out(self.headers),
                },
                "body": self.body,
                "body_encoded": self.body_encoded,
            },
            indent=2,
        )

    def get_body_bytes(self):
        if self.body_encoded:
            return base64.b64decode(self.body_encoded.encode("utf-8"))
        else:
            return self.body.encode("utf-8")

    @classmethod
    def deserialize(cls, json_str: str):
        """Deserialize a JSON string to create an instance of StreamResponse."""
        data = json.loads(json_str.replace("\n", ""))

        # Reconstruct the FastAPI Response object from the serialized data
        response_data = data["response"]
        status_code = int(response_data["status_code"])
        headers = dict(response_data["headers"])

        return cls(
            status_code=status_code,
            headers=headers,
            body=data["body"],
            body_encoded=data["body_encoded"],
            request_query="",
            request_body="",
        )

    async def stream_body(self):
        for piece in self.body:
            print(piece)
            yield piece

    def body_human_readable(self):
        if self.body.startswith("data:"):
            chunks = self.body.strip().splitlines()

            chunk_jsons = []
            for chunk in chunks:
                if chunk.startswith("data: "):
                    # Extract the JSON part after "data: "
                    json_part = chunk[6:]  # Remove "data: "
                    try:
                        # Parse the JSON
                        parsed_json = json.loads(json_part)
                        chunk_jsons.append(parsed_json)
                    except json.JSONDecodeError:
                        continue  # Ignore any malformed JSON

            merged_body = merge_chat_completion_chunks(*chunk_jsons)
            return merged_body
        return ""
