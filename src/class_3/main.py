"""
Interactive command runner for Class 3.
English-only comments.
"""
from src.class_3.logic_layer.naive_bayes.nb_testing import NBTesting
from src.class_3.logic_layer.naive_bayes.nb_training import NBTraining
from src.class_3.logic_layer.vector_spaces.vector_space_training import VSTraining
from src.class_3.logic_layer.vector_spaces.vector_spaces_tests import VSTest


# ============================================================
# HELPERS
# ============================================================

def run_help():
    print("\n================ AVAILABLE COMMANDS ================")
    print("NAIVE BAYES MODULE:")
    print("  nb_train        → Train Naive Bayes with tweet dataset")
    print("  nb_test         → Test Naive Bayes with your own tweet")
    print("")
    print("VECTOR SPACE MODULE:")
    print("  vs_train        → Train Vector Space model")
    print("  vs_test         → Test Vector Space model with a tweet")
    print("")
    print("SYSTEM:")
    print("  help            → Show this help message")
    print("  exit            → Quit program")
    print("====================================================\n")


# ============================================================
# MAIN
# ============================================================



def main():
    print("\n===== Machine Learning Course — Class 3 =====")
    print("Interactive mode. Type a command:")
    run_help()

    while True:
        cmd = input(">> ").strip().lower()

        # ---------- NAIVE BAYES ----------
        if cmd == "nb_train":
            trainer = NBTraining(output_dir="./trained_nb/")
            trainer.run()

        elif cmd == "nb_test":
            tester = NBTesting(models_dir="./trained_nb/")
            tester.run()

        # ---------- VECTOR SPACE ----------
        elif cmd == "vs_train":
            trainer = VSTraining(output_dir="./trained_vs/")
            trainer.run()

        elif cmd == "vs_test":
            tester = VSTest(model_dir="./trained_vs/")
            text = input("Enter tweet: ")
            tester.run_test(text)

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
