import importlib.util
import unittest
import numpy as np
from qbci.adaptation import KernelAdapter, spsa
from qbci.quantum import CircuitSpec
from qbci.data import synthetic_pair
from qbci.experiment import run_pair,ExperimentConfig,DEFAULT_METHODS


class ModelTests(unittest.TestCase):
    def test_all_core_methods_run(self):
        source,target=synthetic_pair(seed=8,trials=48)
        rows,_,_=run_pair(source,target,ExperimentConfig(source_steps=1,target_steps=1),DEFAULT_METHODS)
        self.assertEqual(len(rows),5*len(DEFAULT_METHODS))
        self.assertTrue(all(np.isfinite(r["balanced_accuracy"]) for r in rows))

    def test_subset_freezes_internal_rotations(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(40,4)); y=np.arange(40)%2
        model=KernelAdapter(source_steps=1,target_steps=2).fit_source(x,y)
        initial=model.source_parameters_.copy()
        model.adapt(x[:8],y[:8],"subset")
        np.testing.assert_array_equal(initial[8:],model.parameters_[8:])

    def test_single_class_classifier_replay_works(self):
        rng=np.random.default_rng(4)
        x=rng.normal(size=(40,4)); y=np.arange(40)%2
        model=KernelAdapter(source_steps=1,target_steps=2).fit_source(x,y)
        model.adapt(x[:4],np.zeros(4,int),"subset")
        self.assertEqual(model.update_reason_,"insufficient_class_support")
        self.assertEqual(model.predict(x[:3]).shape,(3,))
        np.testing.assert_array_equal(model.parameters_,model.source_parameters_)

    def test_fixed_optimizer_budget(self):
        calls=[]
        def loss(p): calls.append(1); return float(p@p)
        spsa(loss,np.ones(2),[0,1],[(-2,2),(-2,2)],steps=3)
        self.assertEqual(len(calls),10)

    def test_sampled_projected_and_fidelity(self):
        rng=np.random.default_rng(3)
        x=rng.normal(size=(40,4)); y=np.arange(40)%2
        for kind in ("projected","fidelity"):
            m=KernelAdapter(kind=kind,source_steps=1,target_steps=1,shots=128).fit_source(x,y)
            m.adapt(x[:4],y[:4],"subset")
            self.assertEqual(m.predict(x[:4]).shape,(4,))
            self.assertGreater(m.resources.shots,0)

    @unittest.skipUnless(importlib.util.find_spec("torch"),"PyTorch optional")
    def test_eegnet_bn_frozen(self):
        import torch
        torch.set_num_threads(1)
        from qbci.eegnet import EEGNetAdapter
        source,target=synthetic_pair(seed=2,trials=40,channels=3,sfreq=64)
        m=EEGNetAdapter(source_epochs=1,target_epochs=1).fit_source(source.epochs,source.labels,source.sfreq)
        before={k:v.clone() for k,v in m.model_.state_dict().items() if "running_" in k}
        m.adapt(target.epochs[:4],target.labels[:4])
        after=m.model_.state_dict()
        for k,v in before.items(): torch.testing.assert_close(v,after[k])
        self.assertEqual(m.predict(target.epochs[32:]).shape,(8,))


if __name__=="__main__": unittest.main()
