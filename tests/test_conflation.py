import unittest

import sys
import os
import numpy as np
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(root_dir)
from conflation import conflate_probabilities, fit_conflation_weights, DEFAULT_FLOOR


class TestConflateProbabilities(unittest.TestCase):

    def test_product_and_normalization(self):
        '''TEST 1: well above the floor, conflation is the renormalized elementwise product'''
        a, b = np.array([0.5, 0.3, 0.2]), np.array([0.2, 0.6, 0.2])
        expected = a * b / (a * b).sum()
        np.testing.assert_allclose(conflate_probabilities([a, b]), expected)

    def test_zero_does_not_veto(self):
        '''TEST 2: a class one source gives 0 can still win when the other source is sure of it'''
        classifier = np.array([0.99998, 1e-5, 1e-5])
        markov = np.array([0.0, 0.5, 0.5])
        fused = conflate_probabilities([classifier, markov])
        self.assertEqual(fused.argmax(), 0)
        self.assertEqual(conflate_probabilities([classifier, markov], floor=1e-9).argmax(), 1)
        self.assertTrue(np.isfinite(fused).all())
        self.assertAlmostEqual(fused.sum(), 1.0)

    def test_floor_strength(self):
        '''TEST 3: a tiny floor keeps the old veto behaviour, a larger one weakens it'''
        classifier = np.array([0.98, 0.01, 0.01])
        markov = np.array([1e-6, 0.5, 0.5])
        self.assertEqual(conflate_probabilities([classifier, markov], floor=1e-9).argmax(), 1)
        self.assertEqual(conflate_probabilities([classifier, markov], floor=1e-2).argmax(), 0)

    def test_floor_must_be_positive(self):
        '''TEST 4: floor 0 would allow log(0), so it's rejected'''
        with self.assertRaises(ValueError):
            conflate_probabilities([np.array([0.5, 0.5])], floor=0)

    def test_class_weights(self):
        '''TEST 5: class weights multiply the fused vector'''
        a = np.array([0.5, 0.5])
        np.testing.assert_allclose(conflate_probabilities([a, a], class_weights=np.array([3.0, 1.0])),
                                   [0.75, 0.25])

    def test_fit_weights_with_zeros(self):
        '''TEST 6: weight fitting stays finite when a source gives exact zeros'''
        samples = [(np.array([0.9, 0.1, 0.0]), np.array([0.0, 0.7, 0.3]), 0)] * 5
        weights = fit_conflation_weights(samples, 3, epochs=50, floor=DEFAULT_FLOOR)
        self.assertTrue(np.isfinite(weights).all())
        self.assertGreater(weights[0], weights[2])


if __name__ == '__main__':
    unittest.main()
