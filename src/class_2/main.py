"""
Interactive command runner for Class 2.
English-only comments.
"""
from src.class_2.logic_layer.neural_network.neural_network_tests import NeuralNetworkTests
from src.class_2.logic_layer.neural_network.nn_data_exploration import NNDataExploration
# ---------- OLD ML ----------
from src.class_2.logic_layer.old_ml.algorithm_tests import AlgorithmsTests
from src.class_2.logic_layer.old_ml.data_exploration import DataExploration
from src.class_2.logic_layer.old_ml.algorithm_training import AlgorithmTraining

# ---------- NEURAL NETWORK ----------
from src.class_2.logic_layer.neural_network.neural_network_training import NeuralNetworkTraining




# ============================================================
# HELPERS
# ============================================================

def run_intro():
    print("\n🔹 Class 2: Machine Learning + Neural Networks")
    print("🔹 Environment OK — ready to start.\n")


def run_help():
    print("\n================ AVAILABLE COMMANDS ================")
    print("OLD-ML MODULE:")
    print("  old_intro       → Intro message for OLD ML")
    print("  old_explore     → Explore NON-LINEAR dataset (circles)")
    print("  old_train       → Train OLD ML models (LogReg, SVM, DT, KNN)")
    print("  old_test        → Test OLD ML models (F1, CM, ROC, PR)")
    print("")
    print("NEURAL NETWORK MODULE:")
    print("  nn_explore      → Visual exploration for NN dataset (circles)")
    print("  nn_train        → Train Neural Network + save model + architecture PNG")
    print("  nn_test         → Evaluate NN (confusion matrix, ROC/PR curves, samples)")
    print("")
    print("SYSTEM:")
    print("  help            → Show this help message")
    print("  exit            → Quit program")
    print("====================================================\n")



# ============================================================
# MAIN
# ============================================================

def main():
    print("\n===== Machine Learning Course — Class 2 =====")
    print("Interactive mode. Type a command:")
    run_help()

    while True:
        cmd = input(">> ").strip().lower()

        # ---------- OLD ML ----------
        if cmd == "old_intro":
            run_intro()

        elif cmd == "old_explore":
            DataExploration.run()

        elif cmd == "old_train":
            trainer = AlgorithmTraining(
                csv_path=None,
                target_col="target",
                output_dir="./trained_models_old/"
            )
            trainer.run()

        elif cmd == "old_test":
            tester = AlgorithmsTests(models_dir="./trained_models_old/")
            tester.run()


        # ---------- NEURAL NETWORK ----------
        elif cmd == "nn_explore":

            NNDataExploration.run()

        elif cmd == "nn_train":
            trainer = NeuralNetworkTraining(output_dir="./trained_models_nn/")
            trainer.run()

        elif cmd == "nn_test":
            tester = NeuralNetworkTests(models_dir="./trained_models_nn/")
            tester.run()


        # ---------- SYSTEM ----------
        elif cmd == "help":
            run_help()

        elif cmd == "exit":
            print("Bye!")
            break

        else:
            print("Unknown command. Type 'help'.")



if __name__ == "__main__":
    main()
