import unittest
import numpy as np
from copy import deepcopy

from spectf import utils

class TestUtils(unittest.TestCase):
    """
    :TestUtils:

    The purpose of these test cases are to test the util functions
    """

    def setUp(self): 
        self.dummy_bands = np.arange(start=380, stop=2501, step=10) # should have 212 bands
        self.dummy_spectra = np.random.rand(100, len(self.dummy_bands)) # should be shape (100, 212)

    def test_drop_bands(self):
        def td(drop_wl_ranges=None, nan=False):
            return utils.drop_bands(
                deepcopy(self.dummy_spectra), 
                deepcopy(self.dummy_bands), 
                drop_wl_ranges=drop_wl_ranges, 
                nan=nan
            )
        
        drop_wl_ranges = [
            [380, 410], # 380, 390, 400, 410
            [1270, 1320], # 1270, 1280, 1290, 1300, 1310, 1320
            [2450, 2500], # 2450, 2460, 2470, 2480, 2490, 2500
        ]
        expected_decrease = sum((t[1]-t[0])//10 for t in drop_wl_ranges) + len(drop_wl_ranges)
        spec, bdef = td(drop_wl_ranges=drop_wl_ranges, nan=False)
        self.assertEqual(
            spec.shape, 
            (self.dummy_spectra.shape[0], self.dummy_spectra.shape[-1]-expected_decrease)
        )
        self.assertEqual(len(bdef), len(self.dummy_bands) - expected_decrease)
        for a, b in drop_wl_ranges:
            self.assertNotIn(a, bdef)
            self.assertNotIn(b, bdef)

        spec, bdef = td(drop_wl_ranges=drop_wl_ranges, nan=True)
        self.assertEqual(spec.shape, self.dummy_spectra.shape)
        self.assertEqual(bdef.shape, self.dummy_bands.shape)
        self.assertEqual(np.isnan(spec).sum(), expected_decrease*self.dummy_spectra.shape[0])

    def test_drop_banddef(self):
        drop_wl_ranges = [
            [380, 410], # 380, 390, 400, 410
            [1270, 1320], # 1270, 1280, 1290, 1300, 1310, 1320
            [2450, 2500], # 2450, 2460, 2470, 2480, 2490, 2500
        ]
        b = utils.drop_banddef(deepcopy(self.dummy_bands), drop_wl_ranges)
        expected_decrease = sum((t[1]-t[0])//10 for t in drop_wl_ranges) + len(drop_wl_ranges)
        self.assertEqual(len(b), len(self.dummy_bands) - expected_decrease)
        for a, o in drop_wl_ranges:
            self.assertNotIn(a, b)
            self.assertNotIn(o, b)

        self.assertRaises(TypeError, utils.drop_banddef, deepcopy(self.dummy_bands), [1])

            
if __name__ == "__main__": 
    unittest.main()
