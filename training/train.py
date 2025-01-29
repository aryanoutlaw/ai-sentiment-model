import os
import argparse
import torchaudio
import torch
from tqdm import tqdm
import json
import sys

from meld_dataset import prepare_dataloaders
from models import MultimodalSentimentModel, MultimodalTrainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch',type=int,default=20) 
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--learning-rate',type=float,default=0.001)

    #data directories
    parser.add_argument('--training',type=str,default="dataset/train")
    parser.add_argument('--validation',type=str,default="dataset/dev")
    parser.add_argument('--testing',type=str,default="dataset/test")

    return parser.parse_args()


def main():
    print("Available Audio Backends:")
    print(str(torchaudio.list_audio_backends()))
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        memory_used = torch.cuda.max_memory_allocated() / 1024**3
        print(f"Initial GPU memory used: {memory_used:.2f} GB")

    train_loader, val_loader, test_loader = prepare_dataloaders(
        train_csv=os.path.join(args.train_dir, 'train_sent_emo.csv'),
        train_video_dir=os.path.join(args.train_dir, 'train_splits'),
        dev_csv=os.path.join(args.val_dir, 'dev_sent_emo.csv'),
        dev_video_dir=os.path.join(args.val_dir, 'dev_splits_complete'),
        test_csv=os.path.join(args.test_dir, 'test_sent_emo.csv'),
        test_video_dir=os.path.join(
            args.test_dir, 'output_repeated_splits_test'),
        batch_size=args.batch_size
    )
    print(f"""Training DSV path: {os.path.join(
        args.train_dir, 'train_sent_emo.csv')}""")
    print(f"""Training video directory: {
          os.path.join(args.train_dir, 'train_splits')}""")

    model = MultimodalSentimentModel().to(device)
    trainer = MultimodalTrainer(model, train_loader, val_loader)
    best_val_loss = float('inf')

    metrics_data = {
        "train_losses": [],
        "val_losses": [],
        "epochs": []
    }

    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        train_loss = trainer.train_epoch()
        val_loss, val_metrics = trainer.evaluate(val_loader)

        # Track metrics
        metrics_data["train_losses"].append(train_loss["total"])
        metrics_data["val_losses"].append(val_loss["total"])
        metrics_data["epochs"].append(epoch)

        # Log metrics in SageMaker format
        print(json.dumps({
            "metrics": [
                {"Name": "train:loss", "Value": train_loss["total"]},
                {"Name": "validation:loss", "Value": val_loss["total"]},
                {"Name": "validation:emotion_precision",
                    "Value": val_metrics["emotion_precision"]},
                {"Name": "validation:emotion_accuracy",
                    "Value": val_metrics["emotion_accuracy"]},
                {"Name": "validation:sentiment_precision",
                    "Value": val_metrics["sentiment_precision"]},
                {"Name": "validation:sentiment_accuracy",
                    "Value": val_metrics["sentiment_accuracy"]},
            ]
        }))

        if torch.cuda.is_available():
            memory_used = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Peak GPU memory used: {memory_used:.2f} GB")

        # Save best model
        if val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            torch.save(model.state_dict(), os.path.join(
                args.model_dir, "model.pth"))

    # After training is complete, evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_metrics = trainer.evaluate(test_loader, phase="test")
    metrics_data["test_loss"] = test_loss["total"]

    print(json.dumps({
        "metrics": [
            {"Name": "test:loss", "Value": test_loss["total"]},
            {"Name": "test:emotion_accuracy",
                "Value": test_metrics["emotion_accuracy"]},
            {"Name": "test:sentiment_accuracy",
                "Value": test_metrics["sentiment_accuracy"]},
            {"Name": "test:emotion_precision",
                "Value": test_metrics["emotion_precision"]},
            {"Name": "test:sentiment_precision",
                "Value": test_metrics["sentiment_precision"]},
        ]
    }))

if __name__ == "__main__":
    main()