import sys
sys.path.append('src')

import typer
from pathlib import Path
from onepiece_classify.infer import ImageRecognition 


def predict_image(
    image_path: str = typer.Argument(help="image path", show_default=True),
    # model_path: str = typer.Argument("checkpoint_notebook.pth", help="model path (pth)", show_default=True),
    download: bool = typer.Argument(True, help="True for download the model automatically", show_default=True),
    device: str = typer.Argument("cpu", help="use cuda if your device has cuda", show_default=True)
):
    predictor = ImageRecognition(download=download, device=device)
    result = predictor.predict(image=image_path)
    typer.echo(f"Prediction: {result}")

if __name__ == "__main__":
    typer.run(predict_image)
