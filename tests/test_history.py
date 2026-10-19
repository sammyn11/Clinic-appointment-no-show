"""Check that booking-time history does not see the future."""

import unittest

import pandas as pd

from noshow.features import add_booking_time_history


class HistoryTests(unittest.TestCase):
    def test_history_uses_only_visits_before_the_booking_date(self):
        rows = [
            # Booked 1 May for 10 June, missed. Nothing earlier exists.
            ("p", "a", "2016-05-01", "2016-06-10", 1),
            # Booked 5 May for 20 May, attended. The June visit has not happened.
            ("p", "b", "2016-05-05", "2016-05-20", 0),
            # Booked 25 May for 2 June. The 20 May visit has happened; 10 June has not.
            ("p", "c", "2016-05-25", "2016-06-02", 1),
            # Booked 5 June for 20 June. 20 May and 2 June are known. 10 June is not.
            ("p", "d", "2016-06-05", "2016-06-20", 0),
        ]
        df = pd.DataFrame(rows, columns=["patient_id", "appointment_id", "sched", "appt", "noshow"])
        df["sched_date"] = pd.to_datetime(df["sched"])
        df["appt_date"] = pd.to_datetime(df["appt"])
        out = add_booking_time_history(df).set_index("appointment_id")

        self.assertEqual(out.loc["a", "prior_appointments"], 0)
        self.assertEqual(out.loc["b", "prior_appointments"], 0)
        self.assertEqual(out.loc["c", "prior_appointments"], 1)
        self.assertEqual(out.loc["c", "prior_noshows"], 0)
        self.assertEqual(out.loc["c", "days_since_last"], 5)
        self.assertEqual(out.loc["d", "prior_appointments"], 2)
        self.assertEqual(out.loc["d", "prior_noshows"], 1)
        self.assertAlmostEqual(out.loc["d", "prior_noshow_rate"], 0.5)

    def test_same_calendar_day_is_not_treated_as_history(self):
        df = pd.DataFrame(
            {
                "patient_id": ["p", "p"],
                "appointment_id": ["a", "b"],
                "sched_date": pd.to_datetime(["2016-05-10", "2016-05-10"]),
                "appt_date": pd.to_datetime(["2016-05-10", "2016-05-11"]),
                "noshow": [1, 0],
            }
        )
        out = add_booking_time_history(df).set_index("appointment_id")
        self.assertEqual(out.loc["a", "prior_appointments"], 0)
        self.assertEqual(out.loc["b", "prior_appointments"], 0)


if __name__ == "__main__":
    unittest.main()
