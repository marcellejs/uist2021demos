from glob import glob
import os
import argparse
from marcelle import Remote, Uploader

parser = argparse.ArgumentParser(
    description="Upload training runs to Marcelle's backend"
)
parser.add_argument(
    "--overwrite",
    dest="overwrite",
    action="store_const",
    const=True,
    default=False,
    help="Overwrite existing data on the server",
)


if __name__ == "__main__":
    LOG_DIR = "marcelle-logs"
    ARGS = parser.parse_args()
    uploader = Uploader(
        Remote(
            backend_root="https://marcelle-uist2021-test.herokuapp.com",
            save_format="tfjs",
            source="keras",
        ),
    )
    runs = [d for d in glob(os.path.join(LOG_DIR, "*")) if os.path.isdir(d)]
    for run in runs:
        uploader.upload(run, overwrite=ARGS.overwrite)
