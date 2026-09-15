import inspect
import threading

import modal

from cua.cloud.modal import common, grpo, merge, train
from cua.settings import CLOUD

VOLUME_NAMES = ("cua-trajectories", "cua-outputs", "cua-hf-cache")


class FakeVolume:
    def __init__(self):
        self.commits = 0

    def commit(self):
        self.commits += 1


def test_volumes_and_mounts_construct_without_a_modal_token():
    assert [common.trajectories.name, common.outputs.name, common.hf_cache.name] == list(VOLUME_NAMES)
    assert (common.DATA_MOUNT, common.OUT_MOUNT, common.CODE_ROOT) == ("/data", "/outputs", "/workspace")
    assert common.HF_CACHE_MOUNT.endswith("/.cache/huggingface")
    assert common.BASE_ENV["PYTHONPATH"] == common.CODE_ROOT and common.BASE_ENV["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    assert (common.REPO_ROOT / "pyproject.toml").is_file() and "config.yaml" in common.configs()
    assert common.PROJECT_FILES == ("pyproject.toml",) and common.PROJECT_DIRS == ("constraints",)


def test_the_image_layer_excludes_what_the_rsync_upload_excludes():
    assert common.PACKAGE_IGNORE[:2] == ["**/__pycache__", "**/*.pyc"]
    assert {"**/public_sites_human.json", "**/public_sites_guarded.json"} <= set(common.PACKAGE_IGNORE)


def test_with_package_closes_an_image_with_the_first_party_tree():
    image = common.with_package(modal.Image.debian_slim(), env={"USE_HF": "1"}, files=("corpus_health.json",))
    assert isinstance(image, modal.Image)
    # add_local_dir("cua", …) is the last layer, so the image's own dependency is that mount.
    assert any(type(dep).__name__ == "Mount" for dep in image.deps())


def test_the_train_and_grpo_lanes_share_one_set_of_volume_handles():
    assert train.outputs is common.outputs and grpo.outputs is common.outputs
    assert train.hf_cache is common.hf_cache and grpo.hf_cache is common.hf_cache
    assert train.trajectories is common.trajectories
    assert train.CODE_ROOT == grpo.CODE_ROOT == common.CODE_ROOT
    assert train.OUT_MOUNT == grpo.OUT_MOUNT == common.OUT_MOUNT


def test_each_lane_keeps_its_own_adapter_default():
    grpo_default = inspect.signature(grpo.grpo.get_raw_f()).parameters["sft_adapter"].default
    assert CLOUD.run_name == "sft-v1" and grpo_default == "sft-v1/adapter"
    # merge.py is a standalone script-mode entrypoint whose image mounts no cua package, so it reads its own env.
    assert merge.ADAPTER_NAME == "sft-v2/adapter"


def test_commit_periodically_commits_until_the_event_is_set():
    volume = FakeVolume()
    stop = common.commit_periodically(volume, 0.01)
    assert isinstance(stop, threading.Event)
    deadline = 0
    while volume.commits < 2 and deadline < 500:
        deadline += 1
        stop.wait(0.01)
    assert volume.commits >= 2
    stop.set()
    stop.wait(0.05)
    frozen = volume.commits
    stop.wait(0.05)
    assert volume.commits == frozen
