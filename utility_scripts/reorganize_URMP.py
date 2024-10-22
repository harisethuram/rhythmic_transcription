import os
import csv
import shutil

# -------------------- Configuration -------------------- #

# Path to your CSV file
CSV_FILE = 'metadata/URMP/metadata.csv'  # Replace with your actual CSV file path

# Base directory where 'splitted_data' will be created
BASE_DIR = 'data/URMP_split'

# -------------------------------------------------------- #

def reorganize_folders(csv_file, base_dir):
    """
    Reads the CSV file and moves folders into splitted_data/{split}/ directories.
    """
    # Create the base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    print(f"Created base directory: {base_dir}")

    with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row_num, row in enumerate(reader, start=2):  # Start at 2 considering header
            split = row.get('split')
            folder_path = "/".join(row.get('notes_path').split('/')[:-1])
            # print(folder_path)
            # input()

            if not split or not folder_path:
                print(f"Row {row_num}: Missing 'split' or 'path'. Skipping.")
                continue

            # Define destination directory
            dest_dir = os.path.join(base_dir, split)
            os.makedirs(dest_dir, exist_ok=True)
            print(f"Ensured directory exists: {dest_dir}")

            # Get the folder name from the path
            folder_name = os.path.basename(os.path.normpath(folder_path))
            dest_path = os.path.join(dest_dir, folder_name)

            # Check if source folder exists
            if not os.path.exists(folder_path):
                print(f"Row {row_num}: Source folder does not exist: {folder_path}. Skipping.")
                continue

            # Check if destination already exists to avoid overwriting
            if os.path.exists(dest_path):
                print(f"Row {row_num}: Destination already exists: {dest_path}. Skipping.")
                continue

            try:
                # Move the folder
                shutil.move(folder_path, dest_path)
                print(f"Row {row_num}: Moved '{folder_path}' to '{dest_path}'")
            except Exception as e:
                print(f"Row {row_num}: Failed to move '{folder_path}' to '{dest_path}'. Error: {e}")

if __name__ == "__main__":
    reorganize_folders(CSV_FILE, BASE_DIR)
