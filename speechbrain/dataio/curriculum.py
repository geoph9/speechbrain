import copy
import logging
import tempfile
import time
import random
import os
from itertools import islice
from typing import (
    Any, List, Tuple, Callable, Optional, Dict,
    Union, Sequence
)
# from collections.abc import Iterable
import numpy as np
import torch
from tqdm import tqdm
import speechbrain as sb
from speechbrain.cl.multiproc_sfs import multiproc_score
from speechbrain.dataio.dataset import (
    DynamicItemDataset, 
    FilteredSortedDynamicItemDataset
)
from speechbrain.utils.distributed import run_on_main
from speechbrain.utils.data_pipeline import DataPipeline
from speechbrain.cl.scoring_functions import wada_snr, speech_rate
from speechbrain.utils.checkpoints import (
    register_checkpoint_hooks,
    mark_as_saver,
    mark_as_loader,
)
# from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class CurriculumBase(DynamicItemDataset):
    @classmethod
    def add_random_noise(cls, id_list: List[str], noise_percentage: float = 0.15) -> List[str]:
        """Implementation of uniform mixing where we add some random noise to the
        easy examples. The noise is sampled from the medium and hard examples and
        controls the percentage of easy examples that will be replaced by medium/hard.
        60% of the noise will come from the hard samples and 40% from the medium ones.

        Arguments:
        ----------
        id_list: List[str]
            A list of utterance ids.
        noise_percentage: float
            The percentage of easy examples that will be replaced by medium/hard.
        
        Returns:
        --------
        A list of utterance ids where some of the easy examples have been replaced
        by medium/hard examples.
        """
        assert 0.0 < noise_percentage < 1
        # Step 1: Split list in 3 parts: [easy examples] [medium examples] [hard examples]
        n_ids = len(id_list)
        _n = n_ids // 3
        easy_ids = id_list[:_n]
        medium_ids = id_list[_n:_n*2]
        hard_ids = id_list[_n*2:]
        n_noisy_samples = int(noise_percentage*len(easy_ids))
        # Step 2: 60% of the noise will come from the hard samples and 40% from the medium ones
        n_samples_hard = round(0.6*n_noisy_samples)
        n_samples_med = round(0.4*n_noisy_samples)
        n_samples = n_samples_med + n_samples_hard  # avoid rounding errors, redo sum
        # Step 3: Sample some random ids.
        hard_samples = random.sample(hard_ids, n_samples_hard)
        # take non-common elements, in other words remove the sampled elements from the 
        # list of hard_ids since they will be moved to the "easy" list
        hard_ids = list(set(hard_ids) ^ set(hard_samples))
        medium_samples = random.sample(medium_ids, n_samples_med)
        # Similarly as with the hard ids
        medium_ids = list(set(medium_ids) ^ set(medium_samples))
        # Step 4: Sample an equivalent number of ids from the easy samples.
        #         These ids are the ones that are going to be replaced.
        easy_sample_ids = random.sample(range(len(easy_ids)), n_samples)
        for sample_index in easy_sample_ids:
            if len(hard_samples) > 0:
                # First add all hard samples and then move to the medium ones
                new_val = hard_samples.pop()
                list_to_append = hard_ids  # just a reference
            else:
                new_val = medium_samples.pop()
                list_to_append = medium_ids
            old_val = easy_ids[sample_index]
            easy_ids[sample_index] = new_val
            list_to_append.append(old_val)
        new_ids = easy_ids + medium_ids + hard_ids
        # logger.info(f"Initial id list: {id_list[:20]}\nFinal id list: {out[:20]}")
        assert len(new_ids) == len(id_list), f"{len(new_ids)=} != {len(id_list)=}\n{new_ids=}\n{id_list=}"
        return new_ids
    

    def split_into_k(self, 
      k: int, 
      reverse: Optional[bool] = False, 
      sorting_dict: Optional[dict] = None,
      incremental: Optional[bool] = False,
    ) -> List[np.ndarray]:
        """
        Arguments:
            k: Number of difficulty groups. E.g. if `reverse` is False then the first
               group will contain the easiest examples and the last one the hardest ones.
            reverse: If true then the subsets will be returned by order "hardest to easiest".
            sorting_dict: The dictionary containing utterance ids as keys and scores as values.
            incremental: If true then each consecutive sub-array will also contain the previous 
              samples.
        Returns:
            A list of `k` numpy arrays of equal length. If incremental is True then
            each array A_i will contain A_{i-1} + A_i.
        """
        sorting_dict = sorting_dict or {}
        if len(self.sorting_dict) == 0 and len(sorting_dict) == 0:
            raise ValueError("The class' dictionary is empty, so you need to pass a valid `sorting_dict` argument.")
        sorting_dict = sorting_dict or self.sorting_dict
        sorted_ids = sorted(sorting_dict, key=lambda x: sorting_dict[x], reverse=reverse)
        splitted = np.array_split(sorted_ids, k)
        if not incremental:
            return splitted
        out = [None]*len(splitted)
        out[0] = splitted[0]
        for i, arr in enumerate(splitted[1:]):
            out[i+1] = np.concatenate((out[i], arr), axis=0)
        return out

    def adaptive_pacing(self,
        sorting_dict: dict,
        n_difficulty_groups: int,
        epochs_per_group: int,
        incremental: bool = True,
        noise_percentage: Optional[float] = None,
        normalize: Optional[bool] = True,
        reverse: Optional[bool] = False,
        current_epoch: Optional[int] = 0,
    ):
        """
        Arguments:
            sorting_dict: The sorting dictionary (scores of each utterance).
            n_difficulty_groups: Number of difficulty groups. Check CurriculumDataset.split_into_k
                for more information.
            epochs_per_group: On how many epochs should each group be used for training?
                E.g. if 2, then the easiest group will be used for 2 epochs, then the
                     next group will be used for the next 2 epochs, and so on.
            incremental: If true then each subsequent subset will also contain the easy 
                examples taken from the previous subset. Check CurriculumDataset.split_into_k for more.
            noise_percentage: For noisy CL. Check CurriculumDataset.filtered_sorted_ids and 
                self.add_random_noise for more.
            normalize: Whether or not the sorting dictionary should be normalized. Notice that
                this normalization is IN-PLACE if inplace is True. The same normalization happens in
                CurriculumDataset._curriculum_filtered_ids
            reverse: Descending sorting?
        """
        logger.info(f"Number of difficulty groups (k): {n_difficulty_groups=}, {epochs_per_group=}")
        if not isinstance(sorting_dict, dict) or len(sorting_dict) == 0:
            raise ValueError(f"Invalid sorting dictionary of type: {type(sorting_dict)}.")
        if normalize:
            sorting_dict = self.normalize_dict(sorting_dict)
        paced_sorted_ids = self.split_into_k(
            k=n_difficulty_groups,
            reverse=reverse,
            sorting_dict=sorting_dict,
            incremental=incremental
        )
        tmp_path = "/m/teamwork/t40511_asr/p/curriculum-e2e/startover/test_recipes/lahjoita_puhetta/ASR/seq2seq/exps/tests/"
        with open(os.path.join(tmp_path, "paced_sorted_ids.txt"), "w") as fw:
            for i, el in enumerate(paced_sorted_ids):
                fw.write(f"{i=}:\t {len(el)=} \t[{', '.join(el[:10])}]\n\n\n\n\n")
        logger.info(f'Saved paced indices under {os.path.join(tmp_path, "paced_sorted_ids.txt")}')
        # self.adaptive_pacing_index is a tuple (in the form of a numpy array)
        # whose first element is the index of paced_sorted_ids which we will use,
        # and the second element is the number of epoch that this index has been used.
        # If the second element is greater than epochs_per_group then we move on to the
        # next group.
        logger.info(f"Adaptive pacing index before update: {getattr(self, 'adaptive_pacing_index', None)}")
        if not hasattr(self, "adaptive_pacing_index"):
            paced_ids_index = max(0, current_epoch // epochs_per_group - 1)
            n_usage_epochs = current_epoch % epochs_per_group - 1
            self.adaptive_pacing_index = np.array((paced_ids_index, n_usage_epochs))
        elif self.adaptive_pacing_index[0] >= len(paced_sorted_ids)-1:
            logger.warning(f"The adaptive pacing index has reached the maximum number "
                f"of groups ({self.adaptive_pacing_index}). We will keep increasing the "
                f"number of epochs that this group has been used, though. Is this intentional?")
        current_indices = paced_sorted_ids[self.adaptive_pacing_index[0]]
        logger.info(f"Number of training samples in the current group: {len(current_indices)}")
        # Increase the number of epochs this group has been used for.
        self.adaptive_pacing_index[1] += 1
        # If the number of epochs exceeds the `epochs_per_group` then
        # we move to the next group.
        if self.adaptive_pacing_index[1] >= epochs_per_group and self.adaptive_pacing_index[0] < len(paced_sorted_ids)-1:
            self.adaptive_pacing_index[0] += 1
            self.adaptive_pacing_index[1] = 0
        self.adaptive_pacing_index[0] = min(self.adaptive_pacing_index[0], len(paced_sorted_ids)-1)
        if isinstance(noise_percentage, float) and 0.0 < noise_percentage <= 1.0:
            current_indices = self.add_random_noise(current_indices, noise_percentage)
            logger.info("Added some random noise among the easy examples.")
        logger.info(f"Adaptive pacing index is: {self.adaptive_pacing_index}")
        return FilteredSortedDynamicItemDataset(self, current_indices)


@register_checkpoint_hooks
class CurriculumDataset(CurriculumBase):
    """ A wrapper around `DynamicItemDataset` which will change the way the dataset
        is sorted. In addition, it aims at filtering out the "hard" examples.
    """
    # TODO: Add the curriculum specific method-names here.
    CURRICULUM_DYNAMIC_KEYS = ["loss", "metric_wer", "metric_cer"]
    CURRICULUM_COMPUTABLE_STATIC_KEYS = ["whisper", "speech_rate", "wada_snr"]  # These need a function in order to be computed.
    CURRICULUM_STATIC_KEYS = ["duration"] + CURRICULUM_COMPUTABLE_STATIC_KEYS
    CURRICULUM_KEYS = CURRICULUM_DYNAMIC_KEYS + CURRICULUM_STATIC_KEYS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step = 0
        self.uniform_mixing_perc = kwargs.get('uniform_mixing_perc', None)
        self._sorting = kwargs.get('sorting', None)
        self.output_dir = kwargs.get('output_dir', tempfile.gettempdir())
        self.n_processes = kwargs.get('n_processes', 1)
        self.scoring_function: Callable[[Dict[str, Any]], float] = kwargs.get('scoring_function', None)
        self.cache_path = kwargs.get(
            'cache_path',
            os.path.join(self.output_dir, "curriculum_cache.tsv")
        )
        self.cache: List[str] = self._load_cache()
    
    def _load_cache(self) -> List[str]:
        """Load the cache from a file. The cache is a list of utterance ids that
        have been used in previous epochs.
        """
        if not os.path.exists(self.cache_path):
            return []
        with open(self.cache_path, "r") as f:
            return [line.strip() for line in f]
    
    def _add_duration_key(self):
        # No need to use 'duration' when we have the default implementation.
        if self._sorting in self.CURRICULUM_DYNAMIC_KEYS:
            # Add the audio's duration to the output keys (utterance length)
            original_keys = self.pipeline.output_mapping
            original_keys['duration'] = 'duration'
            original_keys['id'] = 'id'
            sb.dataio.dataset.set_output_keys([self], original_keys,)
    
    @property
    def sorting(self) -> str:
        return getattr(self, "_sorting", None)
    
    @sorting.setter
    def sorting(self, value):
        self._sorting = value
        # If we are on a curriculum method then add the duration key.
        self._add_duration_key()    

    @property
    def is_adaptive(self) -> bool:
        logger.warning("The `is_adaptive` property is not yet implemented. Will return False.")
        return False

    @property
    def accepted_ids(self) -> List[str]:
        # logger.warning("Pacing functions are not yet implemented. We will ignore `accepted_ids` access.")
        return None
    
    def to_cache_or_not_to_cache(self, utt_info: Dict[str, Any], ) -> bool:
        """Given an utterance info dictionary and a scoring function, we will decide whether this
        utterance is worth caching for future retraining.
        """
        is_not_in_cache = utt_info["id"] not in self.cache
        if not is_not_in_cache:
            return False
        score = self.scoring_function(utt_info)
        if self.reverse_cl:
            is_easy = score <= 0.5
            logger.warning("Crude implementation of caching. We will cache all utterances that have a score <= 0.5.")
        else:
            logger.warning("Crude implementation of caching. We will cache all utterances that have a score > 0.5.")
            is_easy = score > 0.5
        if is_easy:
            self.cache.append(utt_info["id"])
        return is_easy
    
    @mark_as_saver
    def save_cache(self, path: str = None):
        """Save the cache to a file. The cache is a list of utterance ids that
        have been used in previous epochs.
        """
        path = path or self.cache_path
        with open(path, "w") as f:
            for utt_id in self.cache:
                f.write(f"{utt_id}\n")
        logger.info(f"Saved cache to {path}.")
    
    @mark_as_loader
    def load_cache(self, path: str = None, end_of_epoch: bool = False, device: str = None):
        """Load the cache from a file. The cache is a list of utterance ids that
        have been used in previous epochs.
        """
        path = path or self.cache_path
        with open(path, "r") as f:
            self.cache = [line.strip() for line in f]
        logger.info(f"Loaded cache from {path}.")
    
    def get_relevant_indices(
        self,
        original_ids: List[str],
    ) -> List[int]:
        """This method accepts a list of utterance ids and returns a list of the indices
        of the utterances that are accepted by the current pacing function. 
        
        For example, if the pacing function of the previous round rejected the utt_ids 
        "id0", "id1", "id2", and the current `original_ids` are 
        ["id0", "id1", "id2", "id3", "id4"]
        then the returned list will be [3, 4].
        Notice that the returned list contains the indices of the accepted ids.

        Args:
            original_ids (List[str]): A list of utterance ids.

        Returns:
            List[int]: A list of indices of the accepted ids.
        """
        if not getattr(self, "accepted_ids", None):
            # If accepted_ids doesn't exist or it is None or it is an empty list
            return list(range(len(original_ids)))
        return [i for i, utt_id in enumerate(original_ids) if utt_id in self.accepted_ids]
    
    def change_score_column(
            self,
            ids: List[str],
            score_values: Union[torch.Tensor, List[float]],
        ):
        """Change the score column of the dataset to the given value. Note that it only
        changes a subset of the dataset, namely the one that is currently being used.

        Arguments
        ---------
        ids : List[str]
            List of utterance ids.
        score_values : torch.Tensor | List[float]
            A tensor of shape (batch_size,) or a list of length `batch_size` containing
            the new score values.
        """
        assert len(ids) == len(score_values)
        for utt_id, score in zip(ids, score_values):
            self.data[utt_id][self.sorting] = score.item()
        return self
    
    def load_scores(self, scores_path: str):
        """Load the scores from a dictionary. The keys of the dictionary should be the
        utterance ids and the values should be the scores.

        Arguments
        ---------
        scores : str
            A path to a tsv file containing the scores.
        """
        logger.info("Loading sorting information from {}".format(scores_path))
        with open(scores_path, "r") as f:
            for line in f:
                utt_id, difficulty_score = line.strip().split("\t")
                self.data[utt_id][self.sorting] = float(difficulty_score)
        # # Add the audio's duration to the output keys (utterance length)
        # original_keys = self.pipeline.output_mapping
        # original_keys[self.sorting] = self.sorting
        # original_keys['id'] = 'id'
        # sb.dataio.dataset.set_output_keys([self], original_keys,)

        # # OLD approach
        # static_keys = list(self.data[self.data_ids[0]].keys()) + ["id"]
        # output_keys = list(self.pipeline.output_mapping.keys())
        # self.pipeline = DataPipeline(static_keys, self.pipeline.dynamic_items)
        # self.set_output_keys(output_keys + [self.sorting])
        return self
    
    def get_sorting_info(self) -> Dict[str, float]:
        """Returns a dictionary containing the sorting information of the dataset.
        The keys are the utterance ids and the values are the sorting values.
        """
        return {utt_id: self.data[utt_id][self.sorting] for utt_id in self.data.keys()}
    
    def save_sorting_info(self, path: str):
        """Save the sorting information of the dataset to a file. The output format
        is tsv where each line contains the utterance id and the sorting value.

        Arguments
        ---------
        path : str
            The path to the file where the sorting information will be saved.
        """
        with open(path, "w") as fw:
            for utt_id in self.data.keys():
                score = self.data[utt_id][self.sorting]
                fw.write(f"{utt_id}\t{score}\n")
        logger.info(f"Saved sorting information to {path}.")
        return self
    
    def add_snr_column(self):
        """Adds the `wada_snr` column to the dataset which contains the wada snr 
        estimation of each utterance (see speechbrain/cl/scoring_functions.py).
        """
        def wada_snr2(utt_info: Dict[str, Any]) -> float:
            sig = sb.dataio.dataio.read_audio(utt_info["wav"])
            return wada_snr(sig)[0]
        return self.add_new_score_function(wada_snr2, "wada_snr")
    
    def add_speech_rate_column(self):
        """Adds the `speech_rate` column to the dataset which contains the speech rate
        of each utterance (see speechbrain/cl/scoring_functions.py).
        """
        def speech_rate2(utt_info: Dict[str, Any]) -> float:
            sig = sb.dataio.dataio.read_audio(utt_info["wav"])
            text = utt_info["wrd"]
            return speech_rate(sig, text)
        return self.add_new_score_function(speech_rate2, "speech_rate")
    
    def add_new_score_function(self, score_function: Callable[[Dict[str, Any]], float], score_name: str):
        """Adds a new score function to the dataset. The score function should accept
        an utterance id and return a float value.

        Arguments
        ---------
        score_function : Callable[Dict[str, Any], float]
            A function that accepts an utterance id and returns a float value.
        score_name : str
            The name of the new score function.
        """
        if score_name in self.data[self.data_ids[0]].keys():
            logger.info(f"The `{score_name}` column already exists. We will not add it again.")
            return self
        logger.info(f"Adding the `{score_name}` column to the dataset. Note that this might take a while.")
        if self.n_processes > 1:
            scores_file = f"{self.output_dir}/{score_name}.tsv"
            if os.path.exists(scores_file):
                logger.info(f"Found existing scores file at {scores_file}. Loading it.")
            else:
                logger.info(f"Using {self.n_processes} processes to compute the scores.")
                multiproc_score(
                    self.data,
                    score_function,
                    n_procs=self.n_processes,
                    out_file=scores_file,
                )
            # Update the dataset with the new scores.
            self.load_scores(scores_file)
        else:
            for utt_id in tqdm(self.data.keys()):
                utt_info = self.data[utt_id]
                # utt_info is of the form: {'wav': 'path/to/wav', 'wrd': 'path/to/wrd', 'id': 'utt_id'}
                self.data[utt_id][score_name] = score_function(utt_info)
        static_keys = list(self.data[self.data_ids[0]].keys()) + ["id"]
        output_keys = list(self.pipeline.output_mapping.keys())
        self.pipeline = DataPipeline(static_keys, self.pipeline.dynamic_items)
        self.set_output_keys(output_keys + [score_name])
        return self

    # This function will be called before each
    def filtered_sorted(self,
        key_min_value: Optional[dict] = {},
        key_max_value: Optional[dict] ={},
        key_test: Optional[dict] = {},
        sort_key: Optional[str] = None,
        reverse: bool = False,
        select_n: int = None,
        save_path: Optional[str] = None,
    ) -> FilteredSortedDynamicItemDataset:
        """Get a filtered and/or sorted version of this, shares static data.

        The reason to implement these operations in the same method is that
        computing some dynamic items may be expensive, and this way the
        filtering and sorting steps don't need to compute the dynamic items
        twice.

        Arguments
        ---------
        key_min_value : dict
            Map from key (in data or in dynamic items) to limit, will only keep
            data_point if data_point[key] >= limit
        key_max_value : dict
            Map from key (in data or in dynamic items) to limit, will only keep
            data_point if data_point[key] <= limit
        key_test : dict
            Map from key (in data or in dynamic items) to func, will only keep
            data_point if bool(func(data_point[key])) == True
        sort_key : None, str
            If not None, sort by data_point[sort_key]. Default is ascending
            order.
        reverse : bool
            If True, sort in descending order.
        select_n : None, int
            If not None, only keep (at most) the first n filtered data_points.
            The possible sorting is applied, but only on the first n data
            points found. Meant for debugging.
        save_path : None, str
            If not None, save the sorted ids to the given path.

        Returns
        -------
        FilteredSortedDynamicItemDataset
            Shares the static data, but has its own output keys and
            dynamic items (initially deep copied from this, so they have the
            same dynamic items available)

        NOTE
        ----
        Temporarily changes the output keys!
        """
        if sort_key == "wada_snr":
            self.add_snr_column()
        elif sort_key == "speech_rate":
            self.add_speech_rate_column()
        elif sort_key in self.CURRICULUM_COMPUTABLE_STATIC_KEYS:
            self.add_new_score_function(self.scoring_function, sort_key)
        # assert sort_key == self.sorting, f"{sort_key=} != {self.sorting=}"
        # keys = list(self.pipeline.output_mapping.keys())
        # self.set_output_keys(list(self.pipeline.output_mapping.keys()) + [sort_key])
        if save_path is not None:
            logger.info(f"Saving ids and scores to {save_path} (before filtering).")
            self.save_sorting_info(save_path)
        # key_max_value should be considered the threshold up to which we allow
        # difficulty scores. I.e. if sorting_dict = {"1": 0.1, "2": 0.8, "3": 0.3}
        # and key_max_value is 0.5 then we will only accept ids "1" and "3".
        filtered_sorted_ids: List[str] = self._filtered_sorted_ids(
            key_min_value, key_max_value, key_test, sort_key, reverse, select_n,
        )
        # Allow uniform mixing regardless of the underlying method (i.e. it can also
        # be used with duration based sorting).
        if self.uniform_mixing_perc:
            logger.info(f"Adding {self.uniform_mixing_perc} random noise to {len(filtered_sorted_ids)} ids.")
            print(filtered_sorted_ids[:10])
            filtered_sorted_ids = self.add_random_noise(filtered_sorted_ids, self.uniform_mixing_perc)
            # random.shuffle(filtered_sorted_ids)
            print(filtered_sorted_ids[:10])
        return FilteredSortedDynamicItemDataset(
            self, filtered_sorted_ids
        )
    
class AdaptiveCLDataset(CurriculumDataset):
    """A dataset that will be used for adaptive curriculum learning.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_duration_key()
        self._adaptive_pacing_index = None
    
    @property
    def accepted_ids(self) -> List[str]:
        raise NotImplementedError("Adaptive pacing is not yet implemented.")