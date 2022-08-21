import unittest
import glyco as gl

TEST_FILE = 'test/data/tiny_sample.csv'

class TestReadFile(unittest.TestCase):
    def test_read(self):
        df = gl.read_from_csv(TEST_FILE, device=gl.Devices.ABBOTT)
        print(df)

class TestGlyco(unittest.TestCase):
    def setUp(self):
        self.glucose_df = gl.read_from_csv(TEST_FILE, device=gl.Devices.ABBOTT)

    def test_derivative(self):
        pass


if __name__ == '__main__':
    unittest.main()
