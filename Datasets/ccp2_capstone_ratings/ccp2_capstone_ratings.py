"""ccp2_capstone_ratings dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import csv

# TODO(ccp2_capstone_ratings): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
This dataset contains mock data of user rankings for different **animes and movies**. This
is based on the MovieLens dataset from TensorFlow and the offline anime dataset from 
manami-project/anime-offline-database.
"""

# TODO(ccp2_capstone_ratings): BibTeX citation
_CITATION = """
"""


class Ccp2CapstoneRatings(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for ccp2_capstone_ratings dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(ccp2_capstone_ratings): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            "media_id": tfds.features.Text(),
            "media_title": tfds.features.Text(),
            "user_id": tfds.features.Text(),
            "user_rating": tf.float32,
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=(None),  # Set to `None` to disable
        disable_shuffling=True,
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(ccp2_capstone_ratings): Downloads the data and defines the splits
    path = dl_manager.extract('../../RawData/clean_ratings.zip')

    # TODO(ccp2_capstone_ratings): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path / 'clean_ratings.csv'),
    }

  def _generate_examples(self, path):
    """Yields examples."""
    # TODO(ccp2_capstone_ratings): Yields (key, example) tuples from the dataset
    with path.open() as csv_file:
      for row in csv.DictReader(csv_file):
        record_id = row["record_id"]
        yield record_id, {
            'media_id': row["media_id"],
            'media_title': row["media_title"],
            'user_id': row["user_id"],
            'user_rating': row["user_rating"],
        }
