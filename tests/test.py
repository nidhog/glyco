import unittest
import ..src.glyco.glucose as gl

TEST_FILE = 'data/tiny_sample.csv'
TEST_FILE_LEN = 4

class TestReadFile(unittest.TestCase):
    def test_read(self):
        df = gl.read_csv(TEST_FILE, device=gl.Devices.abbott)
        self.assertEqual(len(df), TEST_FILE_LEN)
        assert False


class TestGlyco(unittest.TestCase):
    def setUp(self):
        self.glucose_df = gl.read_csv(TEST_FILE, device=gl.Devices.abbott)

    def test_derivative(self):
        pass


if __name__ == '__main__':
    unittest.main()


"""Tests to do:
- Test masking private information achieves purpose.
- 
"""