"""Contains code to spawn multiple processes for scoring functions.
all functions are defined such that they take in a data dictionary
(see `self.data` in `speechbrain.dataio.dataset`), and save a tsv
file containing the results of the scoring function for each utterance
if.
"""

import os
from typing import Any, Callable, Dict
import torch.multiprocessing as mp

import torch
from tqdm import tqdm


mp.set_start_method("spawn", force=True)


def chunk_worker(*chunk):
    chunk, score_func = chunk[0]
    def force_cudnn_initialization():
        s = 32
        dev = torch.device('cuda')
        torch.nn.functional.conv2d(torch.zeros(s, s, s, s, device=dev), torch.zeros(s, s, s, s, device=dev))
    force_cudnn_initialization()
    return {k: score_func(v) for k, v in tqdm(chunk.items())}

def multiproc_score(
        data: Dict[str, Dict[str, Any]],
        score_func: Callable[[Dict[str, Any]], float],
        n_procs: int = 4,
        out_file: str = "results.tsv",
    ) -> None:
    """Spawn multiple processes to run a scoring function on a data dictionary.
    The scoring function must take in a dictionary containing the wav file
    and the ground truth text, and return a float. It is recommended that
    the scoring function be defined in a separate file, and imported here.

    NOTE: If the scoring function uses a GPU, then each process will use
    the same GPU since this is a rather simple implementation. If you want
    to use multiple GPUs, then you will have to implement your own version
    of this function.

    Arguments
    ---------
    data : Dict[[str], Dict[str, Any]]
        The data dictionary. See `self.data` in `speechbrain.dataio.dataset`.
    score_func : Callable[[Dict[str, Any]], float]
        The scoring function to run on a single utterance.
    n_procs : int
        The number of processes to spawn.
    out_file : str
        The path to the output file.
    """

    # Create the pool of processes
    pool = mp.Pool(n_procs)

    # Split the data into {n_procs} chunks
    chunks = []
    chunk_size = len(data) // n_procs
    for i in range(n_procs):
        chunks.append({k: data[k] for k in list(data.keys())[i * chunk_size : (i + 1) * chunk_size]})
    chunks[-1].update({k: data[k] for k in list(data.keys())[(i + 1) * chunk_size :]})
    # Spawn {n_procs} processes, each processing one chunk
    # Make sure the score_func is imported in the worker function
    results = pool.map(chunk_worker, zip(chunks, [score_func] * n_procs))
    # Close the pool
    pool.close()
    pool.join()
    # Merge the results and save them to a file
    results_dict = {}
    for result in results:
        results_dict.update(result)
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))
    with open(out_file, "w") as f:
        for k, v in results_dict.items():
            f.write(f"{k}\t{v}\n")
