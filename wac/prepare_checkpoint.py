"""Convert the published Cosmos Policy weights to the trainer's DCP format."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch.distributed import checkpoint as dcp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    state = torch.load(args.source, map_location="cpu", mmap=True, weights_only=True)
    if not state or any(not key.startswith("net.") for key in state):
        raise ValueError("Expected the published LIBERO checkpoint's flat net.* parameters")

    def validate(directory):
        metadata = dcp.FileSystemReader(directory / "model").read_metadata()
        if metadata.state_dict_metadata.keys() != state.keys():
            raise ValueError("Distributed checkpoint keys do not match the published weights")
        for key, value in state.items():
            if metadata.state_dict_metadata[key].size != value.shape:
                raise ValueError(f"Distributed checkpoint shape mismatch for {key}")

    args.target.parent.mkdir(parents=True, exist_ok=True)
    if args.target.exists():
        validate(args.target)
    else:
        with TemporaryDirectory(dir=args.target.parent) as temporary:
            directory = Path(temporary) / "checkpoint"
            dcp.save(state, checkpoint_id=directory / "model")
            validate(directory)
            directory.rename(args.target)
    print(f"Verified {len(state)} parameter and buffer entries at {args.target}", flush=True)


if __name__ == "__main__":
    main()
