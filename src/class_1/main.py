"""
Interactive command runner for Class 1.
English-only comments.
"""

import sys

def run_intro():
    print("\n🔹 Class 1: Introduction to Machine Learning")
    print("🔹 Environment OK — ready to start.\n")

def run_help():
    print("\nAvailable commands:")
    print("  intro     → Show class introduction")
    print("  exit      → Quit program")
    print("  help      → Show this help message\n")

def main():
    print("\n===== Machine Learning Course — Class 1 =====")
    print("Interactive mode. Type a command:")
    run_help()

    while True:
        cmd = input(">> ").strip().lower()

        if cmd == "intro":
            run_intro()
        elif cmd == "help":
            run_help()
        elif cmd == "exit":
            print("Bye!")
            break
        else:
            print("Unknown command. Type 'help'.")

if __name__ == "__main__":
    main()
