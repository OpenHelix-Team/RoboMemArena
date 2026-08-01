from __future__ import annotations

from .frozen_subprocess import FrozenSubprocessPlugin


class NativePlugin(FrozenSubprocessPlugin):
    """Runs a repository-native evaluator through the same strict subprocess contract."""
