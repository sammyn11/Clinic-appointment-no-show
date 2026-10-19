# Data

The modelling file is not in git. The summative brief says not to commit the raw dataset.

Download it into this folder before training:

```bash
mkdir -p data
curl -L -o data/noshowappointments-kagglev2-may-2016.csv \
  https://raw.githubusercontent.com/ksatola/Medical-Appointments-No-Shows/master/noshowappointments-kagglev2-may-2016.csv
```

The training script also downloads that URL if the file is missing.

Source: Joni Hoppen, "Medical Appointment No Shows", Kaggle, May 2016.
https://www.kaggle.com/datasets/joniarroba/noshowappointments

Licence: CC BY-NC-SA 4.0. This is a student project, not a commercial product. If you redistribute the file, keep the same licence and credit Hoppen / Aquarela Advanced Analytics.

The file has 110,527 appointments and 14 columns. After the cleaning rules in `src/noshow/data.py` (impossible ages, negative lead times) about 110,500 rows remain. That is the table the experiments use.
