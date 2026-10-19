"""Build the local folder that gets uploaded to Google Drive.

The brief wants a folder anyone with the link can view, containing the zipped
raw csv and the two saved models. The zip is gitignored. Uploading the folder
and pasting the link into the report header is a manual step.
"""

from pathlib import Path
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "drive_bundle"
CSV = ROOT / "data" / "noshowappointments-kagglev2-may-2016.csv"
MODELS = ROOT / "models"


def main() -> None:
    if not CSV.exists():
        raise SystemExit(f"missing raw csv: {CSV}. Run the training script or the notebook once.")
    if BUNDLE.exists():
        shutil.rmtree(BUNDLE)
    BUNDLE.mkdir(parents=True)
    zip_path = BUNDLE / "noshowappointments-kagglev2-may-2016.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(CSV, arcname=CSV.name)
    for name in (
        "best_traditional.joblib",
        "best_deep.keras",
        "best_deep.h5",
        "best_deep_preprocess.json",
    ):
        shutil.copy2(MODELS / name, BUNDLE / name)
    note = BUNDLE / "README.txt"
    note.write_text(
        "\n".join(
            [
                "Upload this whole folder to Google Drive.",
                "Set the sharing to: Anyone with the link can view.",
                "Paste that link into the header of report/summative_report.tex and recompile.",
                "",
                "Contents:",
                "- noshowappointments-kagglev2-may-2016.zip  (raw Kaggle file, CC BY-NC-SA 4.0, Hoppen 2016)",
                "- best_traditional.joblib  (E05 random forest, chosen on validation ROC-AUC)",
                "- best_deep.keras and best_deep.h5  (E11 sequential network)",
                "- best_deep_preprocess.json  (scaler and column order for the network)",
                "",
                "Do not commit the zip. data/*.zip and drive_bundle/ are gitignored.",
                "",
            ]
        )
    )
    print(f"wrote {BUNDLE}")
    for path in sorted(BUNDLE.iterdir()):
        print(f"  {path.name:48} {path.stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
