import copy
import unittest
import numpy as np
from qbci.protocol import (prefix_plans, representation_update_allowed, LabelOracle,
                           source_replay_indices, validate_pair)
from qbci.data import synthetic_pair
from qbci.preprocessing import causal_filter_epochs, covariance_epochs, SourceFeatures
from qbci.experiment import run_pair, ExperimentConfig
from qbci.adaptation import KernelAdapter
from qbci.replay import sequential_replay
from qbci.validation import create_lock, canonical_hash, validate_development_pairs


class ProtocolTests(unittest.TestCase):
    def test_same_suffix(self):
        plans = prefix_plans(100)
        for p in plans:
            np.testing.assert_array_equal(p.evaluation, np.arange(32,100))
            np.testing.assert_array_equal(p.calibration, np.arange(p.budget))
            self.assertEqual(len(set(p.calibration) & set(p.evaluation)), 0)

    def test_short_session_rejected(self):
        with self.assertRaises(ValueError): prefix_plans(32)

    def test_unordered_budgets_rejected(self):
        with self.assertRaises(ValueError): prefix_plans(100, (0,8,4))

    def test_one_class_representation_freezes(self):
        self.assertFalse(representation_update_allowed([0,0,0,0]))
        self.assertFalse(representation_update_allowed([0,0,0,1]))
        self.assertTrue(representation_update_allowed([0,0,1,1]))

    def test_oracle_enforces_order_and_budget(self):
        oracle = LabelOracle([0,1,0],1)
        with self.assertRaises(ValueError): oracle.reveal(0)
        with self.assertRaises(ValueError): oracle.score(1,0)
        oracle.score(0,1)
        self.assertEqual(oracle.reveal(0),0)
        with self.assertRaises(ValueError): oracle.reveal(0)
        oracle.score(1,1)
        with self.assertRaises(ValueError): oracle.reveal(1)

    def test_source_replay_is_reproducible(self):
        labels = np.arange(100)%2
        a = source_replay_indices(labels,32,3)
        np.testing.assert_array_equal(a,source_replay_indices(labels,32,3))
        self.assertEqual(np.sum(labels[a]),16)

    def test_chronology_and_identity(self):
        source,target=synthetic_pair(trials=40)
        validate_pair(source,target)
        with self.assertRaises(ValueError): validate_pair(target,source)
        target.subject="another"
        with self.assertRaises(ValueError): validate_pair(source,target)

    def test_causal_filter_future_perturbation(self):
        source,_=synthetic_pair(trials=40)
        x=source.epochs[:1].copy(); changed=x.copy(); changed[:,:,300:] += 1000
        a=causal_filter_epochs(x,source.sfreq)
        b=causal_filter_epochs(changed,source.sfreq)
        np.testing.assert_array_equal(a[:,:,:,:300],b[:,:,:,:300])

    def test_per_trial_filter_has_no_cross_trial_state(self):
        source,_=synthetic_pair(trials=40)
        x=source.epochs[:2].copy(); changed=x.copy(); changed[0] += 100
        np.testing.assert_array_equal(causal_filter_epochs(x,source.sfreq)[1],
                                      causal_filter_epochs(changed,source.sfreq)[1])

    def test_source_basis_immutable(self):
        source,target=synthetic_pair(trials=40)
        cs=covariance_epochs(source.epochs,source.sfreq)
        ct=covariance_epochs(target.epochs,target.sfreq)
        basis=SourceFeatures().fit(cs)
        before=basis.pca_.components_.copy()
        basis.transform(ct); basis.prefix_reference(ct[:4])
        np.testing.assert_array_equal(before,basis.pca_.components_)

    def test_unused_prefix_cannot_change_low_budget_predictions(self):
        source,target=synthetic_pair(seed=4,trials=48)
        changed=copy.deepcopy(target)
        changed.epochs[4:32] *= 50
        changed.labels[4:32] = 1-changed.labels[4:32]
        config=ExperimentConfig(source_steps=1,target_steps=1)
        a=run_pair(source,target,config,["projected_subset"])[1]
        b=run_pair(source,changed,config,["projected_subset"])[1]
        self.assertEqual([r["prediction"] for r in a if r["budget"]<=4],
                         [r["prediction"] for r in b if r["budget"]<=4])

    def test_suffix_labels_cannot_change_predictions(self):
        source,target=synthetic_pair(seed=5,trials=48)
        changed=copy.deepcopy(target); changed.labels[32:] = 1-changed.labels[32:]
        config=ExperimentConfig(source_steps=1,target_steps=1)
        a=run_pair(source,target,config,["projected_subset"])[1]
        b=run_pair(source,changed,config,["projected_subset"])[1]
        self.assertEqual([r["prediction"] for r in a],[r["prediction"] for r in b])

    def test_sequential_score_before_reveal(self):
        rng=np.random.default_rng(3)
        xs=rng.normal(size=(40,4)); ys=np.arange(40)%2
        xt=rng.normal(size=(10,4)); yt=np.arange(10)%2
        model=KernelAdapter(kind="rbf",source_steps=1,target_steps=1).fit_source(xs,ys)
        rows,events=sequential_replay(model,xt,yt,budget=4)
        for i in range(4): self.assertLess(events.index(("score",i)),events.index(("reveal",i)))
        self.assertEqual(rows[0]["labels_available_at_prediction"],0)
        self.assertEqual(rows[-1]["labels_available_at_prediction"],4)

    def test_confirmation_lock_hash_changes(self):
        lock=create_lock({"rbf_classifier": {}},"rbf_classifier",["BNCI:1"],"abc","def")
        self.assertEqual(canonical_hash(lock["protocol"]),lock["sha256"])
        lock["protocol"]["minimum_practical_auc_gain"]=0.01
        self.assertNotEqual(canonical_hash(lock["protocol"]),lock["sha256"])

    def test_openbmi_never_tunes(self):
        source,target=synthetic_pair(trials=40)
        source.dataset=target.dataset="Lee2019_MI"
        with self.assertRaises(ValueError): validate_development_pairs([(source,target)])


if __name__=="__main__": unittest.main()
