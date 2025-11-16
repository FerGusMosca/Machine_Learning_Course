"""
Interactive command runner for Class 1.
English-only comments.
"""

import sys

from src.class_1.logic_layer.algorithm_tests import AlgorithmsTests
from src.class_1.logic_layer.data_exploration import DataExploration
from src.class_1.logic_layer.algorithm_training import AlgorithmTraining
from src.class_1.logic_layer.single_prediction import SinglePrediction


def run_intro():
    print("\n🔹 Class 1: Introduction to Machine Learning")
    print("🔹 Environment OK — ready to start.\n")


def run_help():
    print("\nAvailable commands:")
    print("  intro         → Show class introduction")
    print("  exploredata   → Load + explore dataset")
    print("  trainmodels   → Train 4 ML algorithms + save PKLs")
    print("  testmodels    → Run visual tests (F1, CM, ROC, PR, Predictions)")
    print("  predictone    → Show one patient + model diagnoses")
    print("  exit          → Quit program")
    print("  help          → Show this help message\n")


def main():
    print("\n===== Machine Learning Course — Class 1 =====")
    print("Interactive mode. Type a command:")
    run_help()

    while True:
        cmd = input(">> ").strip().lower()

        if cmd == "intro":
            run_intro()

        elif cmd == "exploredata":
            DataExploration.run()

        elif cmd == "trainmodels":
            trainer = AlgorithmTraining(
                csv_path="breast_cancer.csv",
                target_col="target",
                output_dir="./trained_models/"
            )
            trainer.run()

        elif cmd == "testmodels":
            tester = AlgorithmsTests(models_dir="./trained_models/")
            tester.run()

        elif cmd == "predictone":
            predictor = SinglePrediction(models_dir="./trained_models/")
            predictor.run()

        elif cmd == "help":
            run_help()

        elif cmd == "exit":
            print("Bye!")
            break

        else:
            print("Unknown command. Type 'help'.")


if __name__ == "__main__":
    main()
