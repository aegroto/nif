import statistics
import sys
import os
import json

def main():
    print("Loading parameters...")
    name = sys.argv[1]
    output_file = sys.argv[2]
    results_folder = sys.argv[3]
    filter = None if len(sys.argv) < 5 else sys.argv[4]

    summary = {
        "name": "NIF",
        "type": "inr",
        "results": {
            "config": list(),
            "state_bpp": list(),
            "bpp": list(),
            "psnr": list(),
            "ms-ssim": list(),
            "ssim": list(),
        }
    }

    results = list()

    for root_folder_name in os.listdir(results_folder):
        try:
            if filter and filter not in root_folder_name:
                continue

            root_folder = f"{results_folder}/{root_folder_name}"

            stats = {
                "bpp": list(),
                "state_bpp": list(),
                "psnr": list(),
                "ms-ssim": list(),
                "ssim": list(),
            }

            print(f"Loading stats files in {root_folder}...")
            for root, dirs, files in os.walk(root_folder):
                for file in files:
                    if file == name:
                        path = f"{root}/{file}"
                        file_stats = json.load(open(path, "r"))
                        for key in stats:
                            if key not in file_stats:
                                print(f"WARNING: No field {key} in {path}")
                                continue
                            stats[key].append(file_stats[key])

            print("Calculating and dumping summary...")
            mean_stats = dict()
            for key in stats:
                try:
                    mean_stats[key] = statistics.mean(stats[key])
                except Exception as ex_stats:
                    print(f"Couldn't calculate mean for stat {key}: {ex_stats}")

            mean_stats["config"] = root_folder_name

            if "psnr" not in mean_stats:
                print(f"WARNING: Broken config {root_folder_name}")
                continue

            results.append(mean_stats)
        except Exception as e:
            print(f"WARNING: Could not load {root_folder_name}: {e}")

    results = sorted(results, key = lambda r: r["psnr"])

    for result in results:
        for key in result:
            summary["results"][key].append(result[key])

    json.dump(summary, open(output_file, "w"), indent=4)

if __name__ == "__main__":
    main()

