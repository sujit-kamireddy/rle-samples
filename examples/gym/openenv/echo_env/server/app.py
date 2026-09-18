# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI server for the Echo OpenEnv environment."""

import os

from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from .echo_environment import EchoEnvironment

max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_fastapi_app(
    EchoEnvironment,
    CallToolAction,
    CallToolObservation,
    max_concurrent_envs=max_concurrent,
)
