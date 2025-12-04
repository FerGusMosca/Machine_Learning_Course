"""
Interactive command runner for Class 3.
English-only comments.
"""
from src.class_3.logic_layer.embeddings_cbow.cbow_testing import CBOWTesting
from src.class_3.logic_layer.embeddings_cbow.cbow_training import CBOWTraining
from src.class_3.logic_layer.n_grams.mgram_autocomplete import NGramAutocomplete
from src.class_3.logic_layer.n_grams.trigram_autocomplete import TrigramAutocomplete

# ---------- NAIVE BAYES ----------
from src.class_3.logic_layer.naive_bayes.nb_testing import NBTesting
from src.class_3.logic_layer.naive_bayes.nb_training import NBTraining

# ---------- VECTOR SPACES ----------
from src.class_3.logic_layer.vector_spaces.vector_space_training import VSTraining
from src.class_3.logic_layer.vector_spaces.vector_spaces_tests import VSTest

# ---------- AUTO-CORRECT ----------
from src.class_3.logic_layer.autocorrect.autocorrect import AutoCorrect

# ---------- CBOW EMBEDDINGS ----------


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
    print("  vs_train        → Train Vector Space model (TF-IDF)")
    print("  vs_test         → Test Vector Space model with a tweet")
    print("")
    print("AUTO-CORRECT MODULE:")
    print("  ac_load         → Load autocorrect model (wordfreq)")
    print("  ac_test         → Suggest spelling corrections")
    print("")
    print("AUTOCOMPLETE N-GRAM MODULE:")
    print("  ac_ng_load      → Train BIGRAM autocomplete")
    print("  ac_ng_test      → Suggest next word using bigrams")
    print("")
    print("AUTOCOMPLETE TRIGRAM MODULE:")
    print("  ac_tri_load     → Train TRIGRAM autocomplete")
    print("  ac_tri_test     → Suggest next word using trigrams")
    print("")
    print("CBOW EMBEDDINGS MODULE:")
    print("  cbow_train      → Train CBOW embeddings model")
    print("  cbow_test       → Predict center word using CBOW")
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

    # State for lazy-loaded models
    autocorrect_model = None
    bigram_model = None
    trigram_model = None
    cbow_model_loaded = False   # NEW

    while True:
        cmd = input(">> ").strip().lower()

        # ----------------------------------------------------
        # NAIVE BAYES
        # ----------------------------------------------------
        if cmd == "nb_train":
            NBTraining(output_dir="./trained_nb/").run()

        elif cmd == "nb_test":
            NBTesting(models_dir="./trained_nb/").run()


        # ----------------------------------------------------
        # VECTOR SPACES
        # ----------------------------------------------------
        elif cmd == "vs_train":
            VSTraining(output_dir="./trained_vs/").run()

        elif cmd == "vs_test":
            tester = VSTest(model_dir="./trained_vs/")
            text = input("Enter tweet: ")
            tester.run_test(text)


        # ----------------------------------------------------
        # AUTO-CORRECT
        # ----------------------------------------------------
        elif cmd == "ac_load":
            print("[AC] Loading autocorrect model...")
            autocorrect_model = AutoCorrect()
            print("[AC] Ready.")

        elif cmd == "ac_test":
            if autocorrect_model is None:
                print("[AC] Model not loaded → loading now...")
                autocorrect_model = AutoCorrect()

            word = input("Enter word: ")
            suggestions = autocorrect_model.suggest(word)

            print("\n[AC] Suggestions:")
            for w, d in suggestions:
                print(f"  {w:20s}  dist={d}")


        # ----------------------------------------------------
        # AUTOCOMPLETE BIGRAM
        # ----------------------------------------------------
        elif cmd == "ac_ng_load":
            print("[AC-NG] Training bigram autocomplete...")
            bigram_model = NGramAutocomplete()
            print("[AC-NG] Ready.")

        elif cmd == "ac_ng_test":
            if bigram_model is None:
                print("[AC-NG] Not loaded → loading now...")
                bigram_model = NGramAutocomplete()

            phrase = input("Enter phrase: ")
            suggestions = bigram_model.autocomplete(phrase)

            print(f"\n[AC-NG] Next-word suggestions for '{phrase}':")
            for w, p in suggestions:
                print(f"  {w:20s} prob={p:.4f}")


        # ----------------------------------------------------
        # AUTOCOMPLETE TRIGRAM
        # ----------------------------------------------------
        elif cmd == "ac_tri_load":
            print("[AC-TRI] Training trigram autocomplete...")
            trigram_model = TrigramAutocomplete()
            print("[AC-TRI] Ready.")

        elif cmd == "ac_tri_test":
            if trigram_model is None:
                print("[AC-TRI] Not loaded → loading now...")
                trigram_model = TrigramAutocomplete()

            phrase = input("Enter phrase: ")
            suggestions = trigram_model.autocomplete(phrase)

            print(f"\n[AC-TRI] Trigram suggestions for '{phrase}':")
            for w, p in suggestions:
                print(f"  {w:20s} prob={p:.4f}")


        # ----------------------------------------------------
        # CBOW EMBEDDINGS
        # ----------------------------------------------------
        elif cmd == "cbow_train":
            print("[CBOW] Training embeddings...")
            texts = [
                "the cat sits on the mat",
                "the dog sits on the floor",
                "the cat eats food",
                "the dog eats meat"
            ]

            trainer = CBOWTraining(output_path="./trained_cbow.npz")
            trainer.run()
            cbow_model_loaded = True

        elif cmd == "cbow_test":
            if not cbow_model_loaded:
                print("[CBOW] Model not trained → run cbow_train first.")
                continue

            tester = CBOWTesting(model_path="./trained_cbow.npz")
            tester.run()

            ctx = input("Enter context words (space separated): ").split()
            tester.predict_center_word(ctx)


        # ----------------------------------------------------
        # SYSTEM
        # ----------------------------------------------------
        elif cmd == "help":
            run_help()

        elif cmd == "exit":
            print("Bye!")
            break

        else:
            print("Unknown command. Type 'help'.")


if __name__ == "__main__":
    main()
