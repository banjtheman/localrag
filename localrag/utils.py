import platform
import subprocess


def check_cuda():
    try:
        subprocess.run(
            ["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except FileNotFoundError:
        return False


def check_mps():
    return platform.machine().startswith("arm64") or platform.processor() == "arm"


def get_device_type():
    if check_cuda():
        return "cuda:0"
    elif check_mps():
        return "mps"
    else:
        return "cpu"
