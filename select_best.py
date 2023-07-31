import statistics
import sys
import os
import json

def main():
    results_folder = sys.argv[1]

    for root_folder_name in os.listdir(results_folder):
        try:
            root_folder = f"{results_folder}/{root_folder_name}"

            path = f"{root_folder}/182664/stats.json"
            file_stats = json.load(open(path, "r"))
            bpp = file_stats["state_bpp"]
            psnr = file_stats["psnr"]

            if bpp < 2.9:
                if psnr > 35.5:
                    print(" ### ")
                    print(root_folder)
                    print(file_stats)

        except Exception as e:
            print(f"WARNING: Could not load {root_folder_name}: {e}")

if __name__ == "__main__":
    main()

