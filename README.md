# Missed appointments, flagged at booking time

This is my machine learning summative. The question is narrow on purpose: when a clerk books a clinic visit, can a model tell which bookings are likely to be missed, using only what is already on the booking screen?

I care about that question because missed visits waste a slot that someone else needed, and in a crowded outpatient clinic the people who do turn up then wait longer. Reminder calls and careful overbooking only help if they are aimed at the right bookings. A model that just says "most people attend" is useless for that, even if its accuracy looks high.

The held-out test patients, after a patient-level split, are the numbers I would quote. A random forest (E05) reaches ROC-AUC 0.736. A small Keras network (E11) reaches 0.731. Logistic regression on the booking fields, with no personal history, reaches 0.716. Lead time by itself reaches 0.693, and the rule "flag anyone who is not booked for today" already has F1 0.430. The forest's F1 is 0.451. I would use that gap to order a reminder list. I would not cancel a slot on it, and I would not take these weights out of Vitória and drop them into another clinic. The appointments are Brazilian public data from 2016, used because the decision matches the problem I care about, not because the file is African.

## What is in here

- `notebooks/appointment_noshow.ipynb` — the Colab notebook I submit. `TRAIN = True` fits every experiment.
- `src/noshow/` — the functions the notebook calls, so the cleaning rules are not copied into five cells.
- `scripts/run_experiments.py` — same training as the notebook, if I am on a laptop instead of Colab.
- `experiments/results.csv` — one row per run, including the inner searches that never saw the test patients.
- `figures/` — the plots the report uses.
- `models/` — the traditional model chosen on validation (`.joblib`) and the network (`.keras` and `.h5`).

## Data

Public Kaggle extract, 110,527 appointments, May 2016, neighbourhoods in Vitória, Brazil. Licence CC BY-NC-SA 4.0. The csv is not committed. See `data/README.md`.

The positive class is a no-show (`No-show = Yes` in the file). About one booking in five is missed.

## How I run it

On Colab: upload or open `notebooks/appointment_noshow.ipynb`. Colab only fetches that file, so paste the GitHub URL into `REPO_URL` in the first code cell (once the repo is public) and run from the top. `TRAIN` stays True. The GPU is switched off in that cell because the network AUCs moved when I left it on. After the training cell, set `COPY_TO_DRIVE = True` in the last cell if you want the zipped csv and the model files in My Drive. Sharing ("Anyone with the link can view") is a click in the Drive UI. The notebook cannot do that click.

If Colab asks to restart the runtime after pip, restart, then run all cells again.

On a laptop, Python 3.12, from the repo root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python scripts/run_experiments.py
```

The first run downloads the csv if you have not already. A full run fits several forests and five networks, so give it time on a CPU. Set the same seed as the script (`42`) and you should land close to the table in `experiments/results.csv`. Tiny differences can still show up in the neural nets because TensorFlow on CPU is not perfectly deterministic.

Check the history logic on its own:

```bash
PYTHONPATH=src python -m unittest tests.test_history -v
```

`--fast` fits a patient sample so you can see whether the script crashes. Do not quote those numbers. The sample is too small and I only use it as a smoke test.

Weights & Biases is in offline mode. The run files are under `experiments/wandb/`. TensorBoard event files are under `experiments/tensorboard/`. The csv of results is the table I actually copied into the report.

## How the experiments are organised

Every final number uses a patient-level split: about 68% train, 12% validation, 20% test, with the no-show rate held at about 20% in each. People are not shared across splits.

Features are things the desk would know while booking. Lead time, age, sex, chronic-condition flags, scholarship (Bolsa Família, which I treat as a social-support flag), the clinic neighbourhood, the weekday, and the patient's own earlier visits. A visit only counts as history if it happened on a day before this booking. SMS is left out of the main models and added back in one ablation, because sending a text is a decision, not a description of the patient.

The test patients are not used to pick a tree depth or a learning rate. Those choices use an inner holdout or the validation split. The F1 threshold is chosen on validation and then frozen.

## Models

Traditional: majority baseline, logistic regression, class-weighted logistic regression with a small C search, random forests, histogram gradient boosting at two learning rates, and a linear SVM.

Deep: a small Sequential network, a deeper one with batch norm and dropout, the same network at a higher learning rate, the same network without class weights, and a Functional model that embeds the neighbourhood instead of using a target rate.

## Drive folder

The Drive folder is built locally and not committed:

```bash
PYTHONPATH=src python scripts/make_drive_bundle.py
```

That writes `drive_bundle/` with the zipped raw csv, `best_traditional.joblib`, and the Keras files. Upload that folder with link sharing set to "Anyone with the link can view".

## Repo hygiene

Do not commit `data/*.csv` or a zip of the raw file. Model files are saved because the brief asks for them; the raw appointments stay out. The notebook trains (`TRAIN = True`) and then reloads the joblib and the keras file to check the test AUC against the csv it just wrote.
