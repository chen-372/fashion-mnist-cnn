import argparse
from pathlib import Path
import torch
from src.models.cnn import FashionCNN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--out', type=str, default='outputs/fashion_cnn.onnx')
    args = ap.parse_args()

    device = torch.device('cpu')
    model = FashionCNN()
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    dummy = torch.randn(1, 1, 28, 28)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, dummy, out_path, input_names=['image'], output_names=['logits'], opset_version=12)
    print(f"Exported to {out_path.resolve()}")

if __name__ == '__main__':
    main()
