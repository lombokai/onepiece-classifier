import gdown
from pathlib import Path


def downloader(output_file):
    file_id = "1M1-1Hs198XDD6Xx-kSWLThv1elZBzJ0j"
    prefix = 'https://drive.google.com/uc?/export=download&id='

    url_download = prefix+file_id
    filename = "checkpoint_notebook.pth"
    if Path(output_file).joinpath(filename).exists():
        print("The model has been downloaded")
    else:
        print("Downloading...")
        gdown.download(url_download, output_file)
        print("Download Finish...")
