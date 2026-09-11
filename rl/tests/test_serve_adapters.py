"""serve_adapter accepts several name=path pairs, so one rental exposes every arm the matched comparison needs."""

from remote_train.jobs import parse_serve_adapters


def test_a_bare_path_keeps_the_canonical_served_name():
    assert parse_serve_adapters("models/cua-sft-v3") == {"cua-student-lora": "models/cua-sft-v3"}


def test_named_pairs_map_each_served_name_to_its_path():
    assert parse_serve_adapters("sft-v2=models/cua-sft-v2,sft-v3=output/run/adapter") == {
        "sft-v2": "models/cua-sft-v2",
        "sft-v3": "output/run/adapter",
    }


def test_empty_means_serve_the_base_alone():
    assert parse_serve_adapters("") == {}
    assert parse_serve_adapters(None) == {}
