import sys
import os

import pickle
import hashlib
import logging

import numpy as np
import random
import math
import torch
from torch.utils.data import Dataset

import src.models as models
from src.functional import predict_next_token
import src.data as data


logger = logging.getLogger(__name__)



def _compute_relation_split_ids(relations, split_by, per_relation_split, test_ids, split_by_relations_test_ratio=0.25):
    """Return train/test id lists for the requested split_by strategy."""
    if test_ids is not None:
        train_ids = [idx for idx in range(len(relations)) if idx not in test_ids]
        return train_ids, list(test_ids)

    if split_by == "relations":
        relation_names = [relation.name for relation in relations]
        test_size = round(len(relation_names) * split_by_relations_test_ratio)
        test_relation_names = set(random.sample(relation_names, test_size))
        train_ids = [idx for idx, relation in enumerate(relations) if relation.name not in test_relation_names]
        test_ids = [idx for idx, relation in enumerate(relations) if relation.name in test_relation_names]
    elif split_by == "all_train":
        train_ids = list(range(len(relations)))
        test_ids = list(range(len(relations))) if per_relation_split else []
    else:
        raise NotImplementedError(f"Unknown split_by: {split_by}")

    return train_ids, test_ids


def _split_relation_samples(relations, fraction, seed: int | None = None):
    """Deterministically split relation.samples into train/test slices."""
    rng = random.Random(seed)
    split_map: dict[str, tuple[list, list]] = {}
    for relation in relations:
        samples = list(relation.samples)
        rng.shuffle(samples)
        split_idx = math.ceil(len(samples) * fraction)
        split_map[relation.name] = (samples[:split_idx], samples[split_idx:])
    return split_map



class RelationEmbeddingsDataset(Dataset):
    @classmethod
    def make_splits(
        cls,
        folder_path,
        splits=["train", "test"],
        split_by="relations",
        split_even_if_all_train=False,
        relation_names_file="",
        test_ids=None,
        mt=None,
        use_cache=True,
        cache_folder="cache",
        relation_embeds=False,
        r_dim=None,
        per_relation_split_frac=0.7,
        split_seed: int | None = None,
        decoders_path="decoders/",
        split_by_relations_test_ratio=0.25,
        hparams_path="hparams/",
    ):
        if split_by in ["relations", "all_train"]:
            datasets = {}
            for split in splits:
                datasets[split] = cls(
                    split=split,
                    split_by=split_by,
                    split_even_if_all_train=split_even_if_all_train,
                    folder_path=folder_path,
                    relation_names_file=relation_names_file,
                    test_ids=test_ids,
                    mt=mt,
                    use_cache=use_cache,
                    cache_folder=cache_folder,
                    relation_embeds=relation_embeds,
                    r_dim=r_dim,
                    per_relation_split_frac=per_relation_split_frac,
                    split_seed=split_seed,
                    decoders_path=decoders_path,
                    split_by_relations_test_ratio=split_by_relations_test_ratio,
                    hparams_path=hparams_path
                )

                # Ensure test_ids remain consistent across train and test
                if test_ids is None:
                    test_ids = datasets[split].test_ids

            return datasets
        else:
            raise NotImplementedError

    def __init__(
        self,
        split,
        split_by,
        split_even_if_all_train=False,
        relation_names_file="",
        folder_path=None,
        test_ids=None,
        mt=None,
        use_cache=False,
        cache_folder="cache",
        relation_embeds=False,
        r_dim=None,
        per_relation_split_frac=0.7,
        split_seed: int | None = None,
        decoders_path="decoders/",
        split_by_relations_test_ratio=0.25,
        hparams_path="hparams/"
    ):
        super().__init__()

        self.split = split
        self.split_by = split_by
        self.split_even_if_all_train = split_even_if_all_train
        self.per_relation_split_frac = per_relation_split_frac
        self.split_seed = split_seed
        self.test_ids = test_ids
        self.relation_names_file = relation_names_file
        self.folder_path = folder_path
        self.decoders_path = decoders_path
        self.cache_folder = cache_folder
        self.test_ids = test_ids
        self.relation_embeds = relation_embeds
        self.r_dim = r_dim
        self.mt = mt
        self.split_by_relations_test_ratio = split_by_relations_test_ratio

        self.original_dataset = data.load_dataset(folder_path, hparams_path=hparams_path)

        self.xs, self.ys, self.relations = self.create_dataset(use_cache = use_cache)


    def create_dataset(self, use_cache = False):

        # Try to load cached data, if available.
        cache_key = self.get_dataset_key()
        cache_file = os.path.join(self.cache_folder, cache_key + ".pkl")
        if os.path.exists(cache_file) and use_cache:
            logger.info("Loading dataset from cache...")
            with open(cache_file, "rb") as f:
                xs, ys, relations = pickle.load(f)
        else:
            logger.info("Cache not found or not used. Creating dataset...")

            # create the dataset
            outer_dim = (self.r_dim if self.r_dim is not None else 4096) + 1 # 1 for the bias term. ONE SHOULD DO AN IF HERE!! LLMA+NEO
            xs = []
            ys = []
            relations = []

            with open(self.relation_names_file, "r", encoding="utf-8") as file:
                num_relations = sum(1 for _ in file)

            if self.relation_embeds == "one_hot":
                self.rel_dim = num_relations
            else:
                self.rel_dim = self.r_dim if self.r_dim is not None else 4096 # ALSO here!!! LMA+NEO


            for rel_idx, r in enumerate(open(self.relation_names_file, "r")):
                r = r.strip().lower()
                relation = self.original_dataset.filter(relation_names=[r])[0]
                relations.append(relation)

                embeds = predict_next_token(mt=self.mt, prompt=relation.name, save_embeds=True, layer_index=relation.properties.h_layer)["embeds"]

                x = embeds.clone().cpu()
                logger.debug("x size: %s", x.size())

                if self.relation_embeds == "random":
                    logger.debug("Random vector to relation %s", relation.name)
                    x = torch.randn_like(x)
                elif self.relation_embeds == "one_hot":
                    x = torch.nn.functional.one_hot(torch.tensor(rel_idx), num_classes=num_relations)
                else:
                    logger.debug("Leaving x as is for %s", relation.name)

                xs.append(x)


                filename = self.decoders_path + relation.name.replace(" ", "_") + f"_layer={relation.properties.h_layer}.npy"
                if os.path.exists(filename):
                    y = np.load(filename)
                    zeros = np.zeros((1, y.shape[1]), dtype=y.dtype)
                    y = np.vstack((y, zeros))
                    assert y.shape == (outer_dim, outer_dim)
                else:
                    y = torch.zeros(outer_dim, outer_dim)
                    logger.warning("Decoder matrix missing, using zero: %s", filename)
                    # If the file does not exist, create a zero matrix of the correct shape
                    # This is a fallback and should be handled more gracefully in production code

                ys.append(y)

            xs = np.array(xs)
            xs = xs.reshape(num_relations, self.rel_dim)
            ys = np.array(ys)

            if use_cache:
                 # Save to cache
                if not os.path.exists(self.cache_folder):
                    os.makedirs(self.cache_folder)
                with open(cache_file, "wb") as f:
                    pickle.dump((xs, ys, relations), f)


        train_ids, test_ids = _compute_relation_split_ids(
            relations=relations,
            split_by=self.split_by,
            per_relation_split=self.split_even_if_all_train,
            test_ids=self.test_ids,
            split_by_relations_test_ratio=self.split_by_relations_test_ratio
        )
        self.test_ids = test_ids


        if self.split == 'train':
            if self.split_by != "all_train":
                xs, ys, relations = xs[train_ids], ys[train_ids], [relations[i] for i in train_ids]
        elif self.split == 'test':
            if self.split_by != "all_train":
                xs, ys, relations = xs[test_ids], ys[test_ids], [relations[i] for i in test_ids]
        else:
            raise ValueError(f"Unknown split: {self.split}")


        if self.split_even_if_all_train and self.split_by == "all_train":
            sample_splits = _split_relation_samples(relations, fraction=self.per_relation_split_frac, seed=self.split_seed)

            updated_relations = []
            for relation in relations:
                train_samples, test_samples = sample_splits.get(relation.name, (relation.samples, []))
                if self.split == "train":
                    updated_relations.append(relation.set(samples=train_samples))
                else:
                    updated_relations.append(relation.set(samples=test_samples))
            relations = updated_relations


        logger.info("len train ids: %s", len(train_ids))
        logger.info("len test ids: %s", len(test_ids))
        xs = torch.FloatTensor(xs)
        ys = torch.FloatTensor(ys)

        return xs, ys, relations


    def get_dataset_key(self):
        """
        Generate a unique cache key based on the folder_path, relation_names_file (using its content hash).
        """
        # Read the file content to hash it
        try:
            with open(self.relation_names_file, "r") as f:
                file_content = f.read()
        except Exception as e:
            file_content = ""
            print(f"Warning: Could not read {self.relation_names_file}. Error: {e}")

        # Create a hash of the file content
        hash_obj = hashlib.md5(file_content.encode('utf-8'))
        relation_hash = hash_obj.hexdigest()

        # Combine parameters into a unique key string
        key = (f"relations_folder_path={self.folder_path}_"
               f"relation_names_file_hash={relation_hash}_"
               f"relation_embeds={self.relation_embeds}_"
               f"split={self.split}")
        # Optionally, you can also hash the full key to shorten it
        full_key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        #full_key_hash = key
        return full_key_hash

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx], self.relations[idx]

    def __len__(self):
        return self.xs.shape[0]


def make_dataset_splits(dataset_name, splits, split_by, relation_names_file, folder_path="data/orig", test_ids=None, mt=None, use_cache=False, relation_embeds=None, **kwargs):
    if dataset_name == "relation_embeddings":
        return RelationEmbeddingsDataset.make_splits(
            splits=splits,
            split_by=split_by,
            test_ids=test_ids,
            folder_path=folder_path,
            relation_names_file=relation_names_file,
            mt=mt,
            use_cache=use_cache,
            relation_embeds=relation_embeds,
            **kwargs)
    else:
        raise Exception("Dataset name not defined", dataset_name)
