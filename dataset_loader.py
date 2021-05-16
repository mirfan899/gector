# coding=utf-8
# Lint as: python3

import datasets

logger = datasets.logging.get_logger(__name__)
_URL = "https://github.com/mirfan899/gector_data/raw/main/"
_TRAINING_FILE = "train.txt"
_DEV_FILE = "valid.txt"
_TEST_FILE = "test.txt"


class GectorConfig(datasets.BuilderConfig):
    """BuilderConfig for Gector"""

    def __init__(self, **kwargs):
        """BuilderConfig for Gector.

        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(GectorConfig, self).__init__(**kwargs)


class Gector(datasets.GeneratorBasedBuilder):
    """Gector dataset."""

    BUILDER_CONFIGS = [
        GectorConfig(name="gector config", version=datasets.Version("1.0.0"), description="gector dataset"),
    ]

    def _info(self):
        labels = open("labels.txt").readlines()
        labels = [label.strip() for label in labels]
        return datasets.DatasetInfo(
            description="gector file",
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "gector_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=
                                labels
                        )
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/mirfan899/gector_data/blob/main/",
            citation="MIT",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        urls_to_download = {
            "train": f"{_URL}{_TRAINING_FILE}",
            "dev": f"{_URL}{_DEV_FILE}",
            "test": f"{_URL}{_TEST_FILE}",
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["dev"]}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": downloaded_files["test"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        with open(filepath, encoding="utf-8") as f:
            guid = 0
            tokens = []
            gector_tags = []
            for line in f:
                if line == "" or line == "\n":
                    if tokens:
                        yield guid, {
                            "id": str(guid),
                            "tokens": tokens,
                            "gector_tags": gector_tags,
                        }
                        guid += 1
                        tokens = []
                        gector_tags = []
                else:
                    # gector tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    gector_tags.append(splits[1].rstrip())
            # last example
            yield guid, {
                "id": str(guid),
                "tokens": tokens,
                "gector_tags": gector_tags,
            }
