# Copyright      2023 the University of Edinburgh (Zeyu Zhao)
from typing import List

import k2
import torch
from torch import nn

from .graph_compiler import GraphCompiler


def ctc_k2(log_probs,
           input_lens,
           graph_compiler,
           texts,
           reduction="mean",
           beam_size=10,
           use_double_scores=True,
           is_training=True,
           ):
    input_lens = (input_lens * log_probs.shape[1]).round().int()

    batch_size = log_probs.shape[0]

    supervision_segments = torch.tensor(
        [[i, 0, input_lens[i]] for i in range(batch_size)],
        device="cpu",
        dtype=torch.int32,
    )

    decoding_graph = graph_compiler.compile(texts)
    tids = graph_compiler.lexicon.texts2tids(texts)
    target_lens = torch.tensor([len(t) for t in tids], device=log_probs.device, dtype=torch.long)

    dense_fsa_vec = k2.DenseFsaVec(
        log_probs,
        supervision_segments,
    )
    loss = k2.ctc_loss(
        decoding_graph=decoding_graph,
        dense_fsa_vec=dense_fsa_vec,
        target_lengths=target_lens,
        output_beam=beam_size,
        reduction=reduction,
        use_double_scores=use_double_scores,
    )

    assert loss.requires_grad == is_training

    return loss


def k2_ctc(log_probs, targets, input_lens, target_lens, reduction="mean"):
    """CTC loss implemented with k2. Please make sure that k2 has been installed properly.
    Note that the blank index must be 0 in this implementation.

    Arguments
    ---------
    predictions : torch.Tensor
        Predicted tensor, of shape [batch, time, chars].
    targets : torch.Tensor
        Target tensor, without any blanks, of shape [batch, target_len]
    input_lens : torch.Tensor
        Length of each utterance.
    target_lens : torch.Tensor
        Length of each target sequence.
    reduction : str
        What reduction to apply to the output. 'mean', 'sum', 'none'.
        See k2.ctc_loss for 'mean', 'sum', 'none'.
    """
    input_lens = (input_lens * log_probs.shape[1]).round().int()
    target_lens = (target_lens * targets.shape[1]).round().int()
    batch_size = log_probs.shape[0]

    max_token_id = log_probs.shape[-1] - 1
    ctc_topo = k2.ctc_topo(max_token_id, modified=False,
                           device=log_probs.device)

    # convert targets to k2.FsaVec
    labels = [targets[i, :target_lens[i]].tolist()
              for i in range(len(target_lens))]
    label_fsas = k2.linear_fsa(labels, device=log_probs.device)

    labels_fsas_with_self_loops = k2.remove_epsilon_and_add_self_loops(
        label_fsas)

    labels_fsas_with_self_loops = k2.arc_sort(labels_fsas_with_self_loops)

    graph = k2.compose(ctc_topo, labels_fsas_with_self_loops,
                       treat_epsilons_specially=False)

    assert graph.requires_grad is False

    supervision_segments = torch.tensor(
        [[i, 0, input_lens[i]] for i in range(batch_size)],
        device="cpu",
        dtype=torch.int32,
    )

    dense_fsa_vec = k2.DenseFsaVec(
        log_probs,
        supervision_segments,
    )

    loss = k2.ctc_loss(
        decoding_graph=graph,
        dense_fsa_vec=dense_fsa_vec,
        output_beam=10,
        reduction=reduction,
        target_lengths=target_lens,
    )
    return loss

def mmi_k2(
        log_probs,
        input_lens,
        graph_compiler,
        texts,
        subsampling_factor=1,
        beam_size=10,
        is_training=True,
        den_scale=1.0,
        reduction="mean",
    ):
    input_lens = (input_lens * log_probs.shape[1]).round().int()

    batch_size = log_probs.shape[0]

    supervision_segments = torch.tensor(
        [
            [i, 0, torch.floor(input_lens[i]/subsampling_factor).item()] \
                for i in range(batch_size)
        ],
        device="cpu",
        dtype=torch.int32,
    )

    dense_fsa_vec = k2.DenseFsaVec(
        log_probs,
        supervision_segments,
        allow_truncate=subsampling_factor-1,
    )
    loss_fn = LFMMILoss(
        graph_compiler=graph_compiler,
        den_scale=den_scale,
        beam_size=beam_size,
        reduction=reduction,
    )
    mmi_loss = loss_fn(dense_fsa_vec=dense_fsa_vec, texts=texts, input_lens=input_lens)

    assert mmi_loss.requires_grad == is_training

    return mmi_loss

class LFMMILoss(nn.Module):
    """
    Computes Lattice-Free Maximum Mutual Information (LFMMI) loss.

    TODO: more detailed description
    """

    def __init__(
        self,
        graph_compiler: GraphCompiler,
        den_scale: float = 1.0,
        beam_size: float = 8.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.graph_compiler = graph_compiler
        self.den_scale = den_scale
        self.beam_size = beam_size
        self.reduction = reduction

    def forward(
        self,
        dense_fsa_vec: k2.DenseFsaVec,
        texts: List[str],
        input_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
          dense_fsa_vec:
            It contains the neural network output.
          texts:
            A list of strings. Each string contains space(s) separated words.
          input_lens:
            A 1-D tensor of dtype ``torch.int32`` containing the lengths of
            neural network output. Must have ``input_lens.dim() == 1``.
        Returns:
            Return a scalar loss. It is the sum over utterances in a batch,
            without normalization.
    
        Disclaimer:
            This function is adapted from the `icefall` repository.
            See icefall/icefall/mmi.py for the original source code.
            This is the non-optimized version of the mmi computation.
        """

        num_graphs, den_graphs = self.graph_compiler.compile(texts, replicate_den=True)

        # TODO: pass output_beam as function argument
        num_lats = k2.intersect_dense(
            num_graphs, dense_fsa_vec, output_beam=self.beam_size, max_arcs=2147483600
        )
        den_lats = k2.intersect_dense(
            den_graphs, dense_fsa_vec, output_beam=self.beam_size, max_arcs=2147483600
        )

        num_tot_scores = num_lats.get_tot_scores(log_semiring=True, use_double_scores=True)

        den_tot_scores = den_lats.get_tot_scores(log_semiring=True, use_double_scores=True)

        tot_scores = num_tot_scores - self.den_scale * den_tot_scores

        if self.reduction == "mean":
            # If reduction is mean then we need to divide the loss of
            # each utterance by its length.
            # loss = mmi_loss / input_lens
            loss = -1 * tot_scores / input_lens.to(tot_scores.dtype).to(tot_scores.device)
        else:
            loss = -1 * tot_scores.sum()
        if loss.item() > 2000:
            print(f"==> {tot_scores=}")
            print(f"==> {num_tot_scores=}")  ## it's always the num that is -inf
            print(f"==> {den_tot_scores=}")
            print(f"=========> {num_lats=}")
            print(f"=========> {self.beam_size=}")
        return loss