import logging
import time
from contextlib import contextmanager
from subprocess import PIPE, Popen

import requests


@contextmanager
def start_local_server(data_dir, port, retries=5, sleep=1):
    logging.info(f"Starting server at port {port}...")
    process = Popen(  # noqa: S603
        [  # noqa: S607
            "python",
            "-m",
            "http.server",
            str(port),
            "-d",
            data_dir,
            "--bind",
            "127.0.0.1",
        ],
        stdout=PIPE,
        stderr=PIPE,
    )

    try:
        for _i in range(retries):
            try:
                requests.get(f"http://localhost:{port}", timeout=1)
                logging.info(f"Server started at port {port}")
                yield process
                break
            except requests.exceptions.ConnectionError as e:
                logging.warning(e)
                logging.warning("Server not ready, waiting...")
                time.sleep(sleep)
                pass
    finally:
        process.terminate()
        process.wait()
