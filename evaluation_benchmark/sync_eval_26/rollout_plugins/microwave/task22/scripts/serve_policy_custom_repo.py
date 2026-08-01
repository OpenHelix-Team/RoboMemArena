#!/usr/bin/env python3
"""Serve an OpenPI checkpoint while overriding the LeRobot repo/norm stats path."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import socket

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config name.")
    parser.add_argument("--dir", required=True, help="Checkpoint directory.")
    parser.add_argument("--repo-id", required=True, help="LeRobot repo used for training norm stats.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-prompt", default=None)
    parser.add_argument("--record", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = _config.get_config(args.config)
    train_config = dataclasses.replace(
        train_config,
        data=dataclasses.replace(train_config.data, repo_id=args.repo_id),
    )

    logging.info("Serving checkpoint: %s", args.dir)
    logging.info("Config: %s", args.config)
    logging.info("Overridden LeRobot repo/norm stats path: %s", args.repo_id)

    policy = _policy_config.create_trained_policy(
        train_config,
        args.dir,
        default_prompt=args.default_prompt,
    )
    policy_metadata = policy.metadata
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
