from skin_lesion.cli.train_model import train_model, evaluate_model
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained skin lesion classifier"
    )

    parser.add_argument("--confusion", action="store_true",
                        help="Plot confusion matrix")

    parser.add_argument("--roc", action="store_true",
                        help="Plot ROC curve")

    parser.add_argument("--all-plots", action="store_true",
                        help="Plot confusion matrix and ROC curve")

    parser.add_argument("--metrics", action="store_true",
                        help="Print evaluation metrics")

    return parser.parse_args()

def main():
    args = parse_args()

    evaluation_requested = any([
        args.confusion,
        args.roc,
        args.all_plots,
        args.metrics,
    ])

    if not evaluation_requested:
        print("No evaluation flags provided. Training model...")
        train_model()

    else:
        print("Evaluation mode")
        evaluate_model(
            plot_confusion=args.confusion,
            plot_roc=args.roc,
            plot_all_plots=args.all_plots,
            print_metrics=args.metrics,
        )


if __name__ == "__main__":
    main()
