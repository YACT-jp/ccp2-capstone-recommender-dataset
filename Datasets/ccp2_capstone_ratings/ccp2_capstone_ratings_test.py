"""ccp2_capstone_ratings dataset."""

import tensorflow_datasets as tfds
from . import ccp2_capstone_ratings


class Ccp2CapstoneRatingsTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for ccp2_capstone_ratings dataset."""
  # TODO(ccp2_capstone_ratings):
  DATASET_CLASS = ccp2_capstone_ratings.Ccp2CapstoneRatings
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
