import sys
sys.path.append('src')

import typer
from pathlib import Path
from onepiece_classify.infer import ImageRecognition 


def predict_image(
    image_path: str = typer.Argument(help="image path", show_default=True),
    model_path: str = typer.Argument("checkpoint/checkpoint_notebook.pth", help="model path (pth)", show_default=True)
):
    predictor = ImageRecognition(model_path=model_path)
    result = predictor.predict(image=image_path)
    typer.echo(f"Prediction: {result}")

if __name__ == "__main__":
    typer.run(predict_image)
