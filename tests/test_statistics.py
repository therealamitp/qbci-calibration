import unittest
import numpy as np
from qbci.statistics import (calibration_auc, sustained_target, paired_summary,
                             participant_auc_rows, balanced_accuracy, holm)


class StatisticsTests(unittest.TestCase):
    def test_auc_constant(self):
        self.assertAlmostEqual(calibration_auc([0.7]*5),0.7)

    def test_auc_explicit_weights(self):
        s=np.array([0.51,0.6,0.65,0.7,0.8])
        self.assertAlmostEqual(calibration_auc(s),np.dot(s,[2,4,6,12,8])/32)

    def test_censor_failure(self):
        self.assertIsNone(sustained_target([0.6]*5))

    def test_nonmonotone_curve_no_false_early_crossing(self):
        self.assertEqual(sustained_target([0.6,0.8,0.7,0.8,0.9]),16)

    def test_ba_not_raw_accuracy(self):
        self.assertEqual(balanced_accuracy([0,0,0,1],[0,0,0,0]),0.5)

    def test_no_one_class_eval(self):
        with self.assertRaises(ValueError): balanced_accuracy([0,0],[0,0])

    def test_participant_unit(self):
        rows=[]
        for person in ("A","B"):
            for session in ("2","3"):
                for seed in (0,1,2):
                    for budget in (0,4,8,16,32):
                        rows.append(dict(dataset="D",subject=person,method="M",source_session="1",
                            target_session=session,seed=seed,budget=budget,balanced_accuracy=0.7))
        out=participant_auc_rows(rows)
        self.assertEqual(len(out),2)

    def test_incomplete_curve_rejected(self):
        row=dict(dataset="D",subject="1",method="M",source_session="1",target_session="2",
                 seed=0,budget=0,balanced_accuracy=0.7)
        with self.assertRaises(ValueError): participant_auc_rows([row])

    def test_paired_ci_reproducible(self):
        self.assertEqual(paired_summary([0.01,0.02,-0.01],bootstrap=200),
                         paired_summary([0.01,0.02,-0.01],bootstrap=200))

    def test_holm(self):
        np.testing.assert_allclose(holm([0.01,0.04,0.03]),[0.03,0.06,0.06])


if __name__=="__main__": unittest.main()
