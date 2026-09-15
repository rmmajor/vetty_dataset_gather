"""The one convert+train recipe the Modal and Vast lanes both run, so a flag change cannot land in only one."""

MODULE_ARGS = ("-m", "cua.sft")
# The training lanes' fixed corpus treatment; the Makefile keeps its own literal copy as the user-facing interface.
CONVERT_ARGS = ("--exclude-waste-steps", "--meandering-mode", "subsample")


def convert_command(python, input_dir, output_dir, val_split, extra=()):
    return [
        python,
        *MODULE_ARGS,
        "convert",
        "--input",
        str(input_dir),
        "--output",
        str(output_dir),
        "--val-split",
        str(val_split),
        *CONVERT_ARGS,
        *extra,
    ]


def train_command(python, config, dataset, extra=()):
    return [python, *MODULE_ARGS, "train", "--config", str(config), "--dataset", str(dataset), *extra]
