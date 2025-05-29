import os
import socket
import torch.distributed as dist
import ray
from ray.train._internal.worker_group import WorkerGroup
from ray.train._internal.utils import get_address_and_port
from ray.train.torch.config import (
    _setup_torch_process_group,
    _TorchBackend,
    TorchConfig
)

def _is_free_port(_port: int) -> bool:
    """Check if a port is free."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("", _port))
            return True
        except OSError:
            return False

def _find_port(_range: range) -> int:

    for port in _range:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if _is_free_port(port):
                return port
    raise RuntimeError("No available port found in the specified range.")

class RangedTorchConfig(TorchConfig):
    @property
    def backend_cls(self):
        return RangedTorchBackend

class RangedTorchBackend(_TorchBackend):

    def on_start(self, worker_group: WorkerGroup, backend_config: RangedTorchConfig): # type: ignore[reportIncompatibleMethodOverride]

        if dist.is_available():
            # Set the appropriate training backend.
            if backend_config.backend is None:
                if worker_group.num_gpus_per_worker > 0:
                    backend = "nccl"
                else:
                    backend = "gloo"
            else:
                backend = backend_config.backend

            master_addr, master_port = worker_group.execute_single(
                0, get_address_and_port
            )

            if "MASTER_ADDR" in os.environ:
                master_addr = os.environ["MASTER_ADDR"]

            if "MASTER_PORT" in os.environ:
                if not _is_free_port(int(os.environ["MASTER_PORT"])):
                    raise RuntimeError(
                        f"Port {os.environ['MASTER_PORT']} is not free. "
                        "Please set a different port using the "
                        "MASTER_PORT environment variable."
                    )
                master_port = int(os.environ["MASTER_PORT"])
            elif "MASTER_MIN_PORT" in os.environ and "MASTER_MAX_PORT" in os.environ:
                master_port = _find_port(range(
                    int(os.environ["MASTER_MIN_PORT"]),
                    int(os.environ["MASTER_MAX_PORT"]) + 1,
                ))

            if backend_config.init_method == "env":

                def set_env_vars(addr, port):
                    os.environ["MASTER_ADDR"] = addr
                    os.environ["MASTER_PORT"] = str(port)

                worker_group.execute(set_env_vars, addr=master_addr, port=master_port)
                url = "env://"
            elif backend_config.init_method == "tcp":
                url = f"tcp://{master_addr}:{master_port}"
            else:
                raise ValueError(
                    f"The provided init_method ("
                    f"{backend_config.init_method}) is not supported. Must "
                    f"be either 'env' or 'tcp'."
                )

            setup_futures = []
            for i in range(len(worker_group)):
                setup_futures.append(
                    worker_group.execute_single_async(
                        i,
                        _setup_torch_process_group,
                        backend=backend,
                        world_rank=i,
                        world_size=len(worker_group),
                        init_method=url,
                        timeout_s=backend_config.timeout_s,
                    )
                )
            ray.get(setup_futures)
        else:
            raise RuntimeError("Distributed torch is not available.")