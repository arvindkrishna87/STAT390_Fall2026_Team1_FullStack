from pathlib import Path
import csv

folders = [
    Path("/projects/e32998/STAT390_Fall2026/patches_cases_8_20_33_35_62_102/patches_Callie"),
    Path("/projects/e32998/STAT390_Fall2026/patches_cases_8_20_33_35_62_102/patches_Javi"),
]

for folder in folders:
    # Save each CSV alongside its folder
    output_csv = folder.parent / f"{folder.name}_files.csv"

    file_names = sorted(
        path.name for path in folder.iterdir() if path.is_file()
    )

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename"])
        writer.writerows([name] for name in file_names)

    print(f"Exported {len(file_names)} filenames to {output_csv.resolve()}")
