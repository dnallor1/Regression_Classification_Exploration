
import os
import subprocess

def run_script(script_name):
    """Utility to execute a Python script."""
    try:
        subprocess.run(["python", script_name], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")

def main():
    print("Welcome to the AI Task Manager")
    print("Choose an option:")
    print("1. Train a model")
    print("2. Test a model")
    print("3. Exit")
    choice = input("Enter your choice (1/2/3): ")

    if choice == "1":
        # Run the training script
        if os.path.exists("train_model.py"):
            print("\nStarting model training...")
            run_script("train_model.py")
        else:
            print("Training script 'train_model.py' not found.")
    elif choice == "2":
        # Run the testing script
        if os.path.exists("test_model.py"):
            print("\nStarting model testing...")
            run_script("test_model.py")
        else:
            print("Testing script 'test_model.py' not found.")
    elif choice == "3":
        print("Exiting...")
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()

