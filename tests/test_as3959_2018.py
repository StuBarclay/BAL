"""Numerical regression cases for AS 3959:2018 Method 1 migration."""
import unittest
import numpy as np
from calculate_bal import bal_esti, convo


class TestAS39592018(unittest.TestCase):
    def test_grassland_and_moorland_boundaries(self):
        # Table 2.6: flat/upslope followed by four downslope bands.
        grass = [[5, 6, 10, 14], [5, 7, 11, 16], [6, 8, 13, 19],
                 [7, 10, 15, 22], [8, 11, 17, 25]]
        moor = [[7, 9, 14, 20], [8, 10, 16, 23], [9, 12, 18, 26],
                [10, 13, 20, 29], [11, 15, 23, 33]]
        for veg, rows in [(7, grass), (8, moor)]:
            for slope, thresholds in enumerate(rows, 1):
                for index, distance in enumerate(thresholds):
                    self.assertEqual(bal_esti(veg, distance - .001, slope, 50),
                                     [100, 40, 29, 19][index])
                    self.assertEqual(bal_esti(veg, distance, slope, 50),
                                     [40, 29, 19, 12.5][index])
        self.assertEqual(bal_esti(7, 8, -1, 50), 29)
        self.assertEqual(bal_esti(8, 8, -1, 50), 40)

    def test_cutoffs_and_nodata(self):
        for fdi in [40, 50, 80, 100]:
            self.assertEqual(bal_esti(7, 49.999, 1, fdi), 12.5)
            self.assertEqual(bal_esti(7, 50, 1, fdi), 0)
            self.assertEqual(bal_esti(7, 50, 6, fdi), 0)
            self.assertEqual(bal_esti(1, 99.999, 1, fdi), 12.5)
            self.assertEqual(bal_esti(1, 100, 6, fdi), 0)
            self.assertEqual(bal_esti(0, 0, 6, fdi), 0)
            self.assertEqual(bal_esti(-99, 0, 1, fdi), -99)
            self.assertEqual(bal_esti(1, 0, -99, fdi), -99)
            self.assertEqual(bal_esti(1, 0, 6, fdi), 200)
        self.assertEqual(bal_esti(8, 50, 1, 50), 12.5)
        self.assertEqual(bal_esti(8, 100, 1, 50), 0)

    def test_invalid_inputs(self):
        for args in [(8, 10, 1, 80), (9, 10, 1, 50), (1, -1, 1, 50),
                     (1, float('nan'), 1, 50), (1, 10, 0, 50), (1, 10, 1, 70)]:
            self.assertRaises(ValueError, bal_esti, *args)

    def test_excluded_raster_contributes_low(self):
        zeros = np.zeros((3, 3))
        ones = np.ones((3, 3))
        result = convo('w', zeros, ones, ones, 25, 50)
        np.testing.assert_array_equal(result[:, 0], [-99, -99, -99])
        np.testing.assert_array_equal(result[:, 1:], np.zeros((3, 2)))


if __name__ == '__main__':
    unittest.main()
