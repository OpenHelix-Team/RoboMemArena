import dataclasses
import enum
import logging
import os
import pathlib
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    config: str
    dir: str


@dataclasses.dataclass
class Default:
    pass


@dataclasses.dataclass
class Args:
    env: EnvMode = EnvMode.ALOHA_SIM
    default_prompt: str | None = None
    port: int = 8000
    record: bool = False
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint("pi05_aloha", "gs://openpi-assets/checkpoints/pi05_base"),
    EnvMode.ALOHA_SIM: Checkpoint("pi0_aloha_sim", "gs://openpi-assets/checkpoints/pi0_aloha_sim"),
    EnvMode.DROID: Checkpoint("pi05_droid", "gs://openpi-assets/checkpoints/pi05_droid"),
    EnvMode.LIBERO: Checkpoint("pi05_libero", "gs://openpi-assets/checkpoints/pi05_libero"),
}


def bind_checkpoint_assets(train_config: _config.TrainConfig, checkpoint_dir: pathlib.Path) -> _config.TrainConfig:
    data_factory = train_config.data
    configured_norm = os.environ.get("VLA_NORM_FILE", "").strip()
    if configured_norm:
        norm_file = pathlib.Path(configured_norm).resolve()
        assets_dir = norm_file.parent.parent
        asset_id = norm_file.parent.name
    else:
        asset_id = data_factory.assets.asset_id
        if asset_id is None:
            repo_id = data_factory.repo_id
            if not repo_id:
                raise ValueError("Cannot derive asset_id because the data config has no repo_id.")
            repo_path = pathlib.Path(repo_id)
            asset_id = repo_path.name if repo_path.is_absolute() else str(repo_id)
        assets_dir = checkpoint_dir / "assets"
        norm_file = assets_dir / asset_id / "norm_stats.json"
    if not norm_file.is_file():
        raise FileNotFoundError(f"Self-contained checkpoint norm is missing: {norm_file}")

    assets = dataclasses.replace(
        data_factory.assets,
        assets_dir=str(assets_dir),
        asset_id=asset_id,
    )
    return dataclasses.replace(train_config, data=dataclasses.replace(data_factory, assets=assets))


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    checkpoint = DEFAULT_CHECKPOINT.get(env)
    if checkpoint is None:
        raise ValueError(f"Unsupported environment mode: {env}")
    return _policy_config.create_trained_policy(
        _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
    )


def create_policy(args: Args) -> _policy.Policy:
    match args.policy:
        case Checkpoint():
            checkpoint_dir = pathlib.Path(args.policy.dir).resolve()
            train_config = bind_checkpoint_assets(_config.get_config(args.policy.config), checkpoint_dir)
            logging.info(
                "Using self-contained checkpoint assets: %s",
                train_config.data.assets.assets_dir,
            )
            return _policy_config.create_trained_policy(
                train_config,
                checkpoint_dir,
                default_prompt=args.default_prompt,
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
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
    main(tyro.cli(Args))
