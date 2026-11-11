import unittest
import numpy as np
from qbci.quantum import (CircuitSpec, states, initial_parameters, local_expectations,
                         product_expectations, rotation, apply_single, fidelity,
                         sampled_expectations, sampled_fidelity)
from qbci.kernels import rbf, alignment, PSDRepair


class QuantumTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)
        self.x = self.rng.normal(size=(12, 4))
        self.spec = CircuitSpec()
        self.p = initial_parameters(self.spec, 4)
        self.v = states(self.x, self.p, self.spec)

    def test_normalization(self):
        np.testing.assert_allclose(np.sum(abs(self.v)**2, axis=1), 1, atol=1e-12)

    def test_parameter_and_gate_counts(self):
        self.assertEqual(self.spec.n_parameters, 20)
        self.assertEqual(self.spec.gates, {"one_qubit": 28, "two_qubit": 6})

    def test_fidelity_gram(self):
        k = fidelity(self.v)
        np.testing.assert_allclose(k, k.T)
        np.testing.assert_allclose(np.diag(k), 1, atol=1e-12)
        self.assertGreater(np.linalg.eigvalsh(k).min(), -1e-10)

    def test_shared_terminal_unitary_cancels(self):
        z = self.rng.normal(size=(16,16)) + 1j*self.rng.normal(size=(16,16))
        unitary = np.linalg.qr(z)[0]
        np.testing.assert_allclose(fidelity(self.v @ unitary.T), fidelity(self.v), atol=1e-12)

    def test_interleaved_parameters_change_kernel(self):
        perturbed = self.p.copy(); perturbed[10] += 0.4
        difference = np.max(abs(fidelity(states(self.x, perturbed, self.spec))-fidelity(self.v)))
        self.assertGreater(difference, 1e-4)

    def test_encoding_parameters_change_kernel(self):
        for index in range(8):
            p = self.p.copy(); p[index] += 0.2
            self.assertGreater(np.max(abs(fidelity(states(self.x, p, self.spec))-fidelity(self.v))), 1e-5)

    def test_terminal_local_rotations_preserve_projected_distances(self):
        v = self.v.copy()
        for wire in range(4):
            for axis in ("x", "y", "z"):
                v = apply_single(v, rotation(axis, self.rng.uniform(-2, 2)), wire, 4)
        np.testing.assert_allclose(rbf(local_expectations(v, 4)),
                                   rbf(local_expectations(self.v, 4)), atol=1e-12)

    def test_product_classical_twin(self):
        spec = CircuitSpec(entangle=False)
        np.testing.assert_allclose(product_expectations(self.x, self.p, spec),
                                   local_expectations(states(self.x, self.p, spec), 4), atol=1e-12)

    def test_shot_convergence(self):
        sample = sampled_expectations(self.v[:2], 4, 100000, self.rng)
        np.testing.assert_allclose(sample, local_expectations(self.v[:2],4), atol=0.015)

    def test_joint_shots_preserve_bell_correlations(self):
        bell = np.array([[1,0,0,1]], complex)/np.sqrt(2)
        sample = sampled_expectations(bell, 2, 128, self.rng)
        self.assertEqual(sample[0,2], sample[0,5])

    def test_finite_fidelity_symmetric(self):
        k = sampled_fidelity(self.v, 128, self.rng)
        np.testing.assert_equal(k, k.T)
        np.testing.assert_equal(np.diag(k), 1)

    def test_projected_psd(self):
        self.assertGreater(np.linalg.eigvalsh(rbf(local_expectations(self.v,4))).min(), -1e-10)

    def test_alignment_one_class(self):
        self.assertEqual(alignment(np.eye(4), np.zeros(4)), 0)

    def test_alignment_weighted_bound(self):
        value = alignment(rbf(self.x), np.arange(12)%2, np.arange(1,13))
        self.assertLessEqual(abs(value), 1+1e-12)

    def test_training_only_psd_extension(self):
        k = np.array([[1,2],[2,1.]])
        repair = PSDRepair(); positive = repair.fit_transform(k)
        self.assertGreater(np.linalg.eigvalsh(positive).min(), -1e-12)
        np.testing.assert_allclose(repair.transform(k), positive)


if __name__ == "__main__":
    unittest.main()
