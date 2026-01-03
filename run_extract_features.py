from skin_lesion.config.config import (DATA_DIR, FEATURES_DIR, IMAGE_SIZE)
from skin_lesion.features.io import save_features
from skin_lesion.preprocessing.io import load_dataset
import argparse
from skin_lesion.cli.extract_features import extract_features, display_superpixels, print_features

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Skin lesion feature extraction"
    )

    parser.add_argument(
        "--display-superpixels",
        action="store_true",
        help="Display superpixels overlay for the first sample",
    )

    parser.add_argument(
        "--print-features",
        action="store_true",
        help="Print extracted feature vectors to stdout",
    )

    args = parser.parse_args()

    # Optional: print features
    if args.print_features:
        print_features()
        return

    # Default behavior: save features
    dataset = load_dataset(DATA_DIR, IMAGE_SIZE)
    print(f"Loaded {len(dataset)} samples")

    # Optional: visualize superpixels
    if args.display_superpixels:
        display_superpixels(dataset)
        return

    X, ids = extract_features(dataset)
    save_features(FEATURES_DIR, X, ids)

if __name__ == "__main__":
    main()
