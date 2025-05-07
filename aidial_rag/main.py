# Init logging before importing anything else to be able to log
# the init process of other dial components
# ruff: noqa: E402
from aidial_rag.log_config import init_logging_and_telemetry

init_logging_and_telemetry()


import uvicorn
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

from aidial_rag.app import create_app
from aidial_rag.app_config import AppConfig

app = create_app(app_config=AppConfig())

# Since we run init telemetry before the app is created,
# we need to explicitly pass the app to the instrumentor
FastAPIInstrumentor.instrument_app(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)  # noqa: S104
