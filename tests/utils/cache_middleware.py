import base64
import gzip
import json
import logging
import mimetypes
import warnings
from pathlib import Path
from typing import Dict

import httpx
import pytest
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from pydantic.dataclasses import dataclass

from tests.utils.cache_response import CacheResponse
from tests.utils.llm_cache import LlmCache, get_cache_key

llm_cache = None

WARNING_MESSAGE = (
    "No cached value found, this means that something was changed in the logic, "
    "so cached LLM responses should be renewed."
)

FAILURE_MESSAGE = (
    "There is no response found in cache, use environment variable REFRESH=True to update cache "
    "and commit to repository"
)

logger = logging.getLogger(__name__)


@dataclass
class CacheMiddlewareConfig:
    dial_core_host: str = "localhost:8080"
    dial_core_api_key: str = "dial_api_key"
    base_path: str = "./tests"
    module_name: str = ""
    test_name: str = ""
    refresh: bool = False


def _get_accept_encoding(headers):
    return headers.get("accept-encoding") or ""


class CacheMiddlewareApp(FastAPI):
    llm_cache: LlmCache

    def get_cache_files(self, prefix):
        return self.llm_cache.get_by_prefix(prefix)

    def __init__(self, app_config: CacheMiddlewareConfig):
        self.dial_core_host = app_config.dial_core_host
        self.dial_core_api_key = app_config.dial_core_api_key
        self.base_path = app_config.base_path
        self.module_name = app_config.module_name
        self.test_name = app_config.test_name
        self.refresh = app_config.refresh
        self.base_path = Path(self.base_path)
        self.llm_cache = LlmCache(
            cache_dir=f"{self.base_path / 'cache'}", enable_cache=True
        )
        self.target_url = f"http://{self.dial_core_host}"

        self.unused_cache = self.get_cache_files(
            f"{self.module_name}/{self.test_name}"
        )

        super().__init__()
        self.router = APIRouter()
        self.register_routes()

    async def send_post(self, body, headers, query_params, request):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url=f"{self.target_url}{request.url.path}",
                params=query_params,
                content=body,
                headers=headers,
                timeout=600,
            )
            cache_response = CacheResponse.from_response(
                response, response.content, str(query_params), body.decode()
            )
        return cache_response

    async def send_get(self, headers, request):
        async with httpx.AsyncClient() as client:
            logger.info(
                f"No cache for GET request. Send to {self.target_url}{request.url.path}"
            )
            response = await client.get(
                f"{self.target_url}{request.url.path}",
                params=request.query_params,
                headers=headers,
                timeout=600,
            )
            content = CacheResponse.from_response(
                response, response.content, request, ""
            )
        return content

    def handle_metadata_request(self, path) -> CacheResponse:
        parts = path.split("/")
        bucket = parts[0]  # This is the bucket value
        file_name = parts[1]
        file_path = f"{self.base_path}/data/{file_name}"
        try:
            with open(file_path, "rb") as f:
                content_length = len(f.read())

            content_type = mimetypes.guess_type(file_name)[0]

            metadata = {
                "name": file_name,
                "parentPath": None,
                "bucket": bucket,
                "url": f"/{path}",
                "nodeType": "ITEM",
                "resourceType": "FILE",
                "contentLength": content_length,
                "contentType": content_type,
            }

            metadata_headers = {
                "content-type": "application/json",
                "content-encoding": "gzip",
            }
            return CacheResponse(
                200,
                metadata_headers,
                json.dumps(metadata),
                "",
                request_query=path,
                request_body="",
            )
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            logger.exception(f"Error processing metadata for {file_name}: {e}")
            raise

    def handle_file_request(self, filename) -> CacheResponse:
        file_path = f"{self.base_path}/data/{filename}"

        metadata_headers = {
            "content-type": mimetypes.guess_type(filename)[0],
            "content-encoding": "gzip",
        }

        try:
            with open(file_path, "rb") as file:
                file_bytes = file.read()
                encoded_bytes = base64.b64encode(file_bytes).decode("utf-8")
                return CacheResponse(
                    200,
                    metadata_headers,
                    "",
                    encoded_bytes,
                    request_query=filename,
                    request_body="",
                )
        except Exception as e:
            logger.exception(f"Error reading file {filename}: {e}")
            raise

    async def handle_post(self, request: Request):
        cache_prefix = f"{self.module_name}/{self.test_name}"
        # Custom handling for POST requests can be added here
        logger.info(f"Receive POST request to {request.url.path}")
        query_params = request.query_params
        path = request.url.path
        body = await request.body()
        headers = await self.get_headers(request)

        input_str = "$".join([path, str(query_params), str(body)])
        cache_response, cache_response_path = self.llm_cache.get(
            input_str, prefix=cache_prefix
        )
        if cache_response is None:
            warnings.warn(WARNING_MESSAGE, stacklevel=2)
            if self.refresh:
                logger.info(
                    f"No cache for POST request. Send to {self.target_url}{request.url.path}"
                )
                cache_response = await self.send_post(
                    body, headers, query_params, request
                )
                self.llm_cache.set(
                    input_str, cache_response, prefix=cache_prefix
                )
            else:
                pytest.fail(
                    f"{FAILURE_MESSAGE}::{get_cache_key(input_str)}::{input_str}"
                )
        else:
            self.mark_as_used(cache_response_path)
        if (
            "content-encoding" in cache_response.headers.keys()
            and "gzip" in cache_response.headers["content-encoding"]
        ):
            body = gzip.compress(cache_response.get_body_bytes())
        else:
            body = cache_response.body
            # return response
        result_response = Response(
            content=body,
            status_code=cache_response.status_code,
            media_type=cache_response.headers.get("content-type"),
            headers=cache_response.headers,
        )
        return result_response

    async def get_headers(self, request: Request) -> Dict[str, str | None]:
        allowed_headers = [
            "accept-encoding",
            "accept",
            "connection",
            "content-type",
            "api-key",
        ]
        headers = {
            key: request.headers.get(key)
            for key in request.headers.keys()
            if key in allowed_headers
        }
        headers["host"] = self.dial_core_host
        headers["api-key"] = self.dial_core_api_key
        return headers

    def register_routes(self):
        @self.get("/v1/metadata/files/{file_path:path}")
        async def get_metadata(request: Request, file_path: str):
            headers: dict = await self.get_headers(request)
            content = self.handle_metadata_request(file_path)
            if "gzip" in _get_accept_encoding(headers):
                body = gzip.compress(content.get_body_bytes())
            else:
                body = content.body
            return Response(
                content=body,
                status_code=content.status_code,
                headers=content.headers,
            )

        @self.get("/v1/files/{bucket}/{file_path:path}")
        async def get_file_content(request: Request, file_path: str):
            headers = await self.get_headers(request)
            content = self.handle_file_request(file_path)
            if "gzip" in _get_accept_encoding(headers):
                body = gzip.compress(content.get_body_bytes())
            else:
                body = content.body
            return Response(
                content=body,
                status_code=content.status_code,
                headers=content.headers,
            )

        @self.post("/{path:path}")
        async def post(request: Request, path: str):
            return await self.handle_post(request)

        @self.get("/v1/deployments/{deployment_name}/limits")
        async def get_deployment_limits(deployment_name: str):
            return {
                "minuteTokenStats": {"total": 60000, "used": 0},
                "dayTokenStats": {"total": 1000000, "used": 0},
            }

        @self.get("/health")
        async def health_check():
            return {"status": "ok"}

    def cleanup(self):
        if self.refresh and len(self.unused_cache) > 0:
            self.llm_cache.cleanup(self.unused_cache)

    def mark_as_used(self, file_path):
        if file_path in self.unused_cache:
            self.unused_cache.remove(file_path)
            logger.info(f"Marked as used: {file_path}")
