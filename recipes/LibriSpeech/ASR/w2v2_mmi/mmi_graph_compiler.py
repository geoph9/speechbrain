import logging
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import k2
import torch
from speechbrain.k2_integration.graph_compiler import CtcTrainingGraphCompiler
from speechbrain.k2_integration.lexicon import Lexicon


logger = logging.getLogger(__name__)


class MmiTrainingGraphCompiler(CtcTrainingGraphCompiler):
    def __init__(
        self,
        lexicon: Lexicon,
        device: torch.device,
        oov: str = "<UNK>",
        need_repeat_flag: bool = False,
        G_path: str = None,
        P_path: str = None,
        rescoring_lm_path: Union[Path, str] = None,
        sos_id: int = 1,
        eos_id: int = 1,
    ):
        """
        Args:
          lexicon:
            It is built from `data/lang/lexicon.txt`.
          device:
            The device to use for operations compiling transcripts to FSAs.
          oov:
            Out of vocabulary word. When a word in the transcript
            does not exist in the lexicon, it is replaced with `oov`.
          need_repeat_flag: bool
            If True, will add an attribute named `_is_repeat_token_` to ctc_topo
            indicating whether this token is a repeat token in ctc graph.
            This attribute is needed to implement delay-penalty for phone-based
            ctc loss. See https://github.com/k2-fsa/k2/pull/1086 for more
            details. Note: The above change MUST be included in k2 to open this
            flag.
          G_path: str
            Path to the language model FST to be used in the decoding-graph creation.
            If None, then we assume that the language model is not used.
          P_path: str,
            Path to the language model FST to be used in the numerator graph creation.
            If None, then we assume that the language model is not used.
          rescoring_lm_path: Path | str
            Path to the language model FST to be used in the rescoring of the decoding
            graph. If None, then we assume that the language model is not used.
          sos_id: int
            ID of the start-of-sentence token.
          eos_id: int
            ID of the end-of-sentence token.
        """
        super().__init__(
            lexicon=lexicon,
            device=device,
            oov=oov,
            need_repeat_flag=need_repeat_flag,
            G_path=G_path,
            rescoring_lm_path=rescoring_lm_path,
        )
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.P_path = P_path
        self._p = None

        self.build_ctc_topo_P()

    @property
    def P(self) -> k2.Fsa:
        if self._p is not None:
            return self._p
        with open(self.P_path) as f:
            # P is not an acceptor because there is
            # a back-off state, whose incoming arcs
            # have label #0 and aux_label 0 (i.e., <eps>).
            self._p = k2.Fsa.from_openfst(f.read(), acceptor=False)
        return self._p

    def build_ctc_topo_P(self):
        """Built ctc_topo_P, the composition result of
        ctc_topo and P, where P is a pre-trained bigram
        word piece LM.
        """
        # Note: there is no need to save a pre-compiled P and ctc_topo
        # as it is very fast to generate them.
        logging.info(f"Loading P from lexicon ({self.P_path})")
        P = self.P
        
        first_token_disambig_id = self.lexicon.token_table["#0"]

        # P.aux_labels is not needed in later computations, so
        # remove it here.
        del P.aux_labels
        # CAUTION: The following line is crucial.
        # Arcs entering the back-off state have label equal to #0.
        # We have to change it to 0 here.
        labels = P.labels.clone()
        labels[labels >= first_token_disambig_id] = 0
        P.labels = labels

        P = k2.remove_epsilon(P)
        P = k2.arc_sort(P)
        P = P.to(self.device)
        # Add epsilon self-loops to P because we want the
        # following operation "k2.intersect" to run on GPU.
        P_with_self_loops = k2.add_epsilon_self_loops(P)

        max_token_id = max(self.lexicon.tokens)
        logging.info(
            f"Building ctc_topo (modified=False). max_token_id: {max_token_id}"
        )

        ctc_topo_inv = k2.arc_sort(self.ctc_topo.invert_())

        logging.info("Building ctc_topo_P")
        ctc_topo_P = k2.intersect(
            ctc_topo_inv, P_with_self_loops, treat_epsilons_specially=False
        ).invert()

        self.ctc_topo_P = k2.arc_sort(ctc_topo_P)
        logging.info(f"ctc_topo_P num_arcs: {self.ctc_topo_P.num_arcs}")

    def compile(
        self, texts: Iterable[str], replicate_den: bool = True
    ) -> Tuple[k2.Fsa, k2.Fsa]:
        """Create numerator and denominator graphs from transcripts
        and the bigram phone LM.

        Args:
          texts:
            A list of transcripts. Within a transcript, words are
            separated by spaces. An example `texts` is given below::

                ["Hello icefall", "LF-MMI training with icefall using k2"]

          replicate_den:
            If True, the returned den_graph is replicated to match the number
            of FSAs in the returned num_graph; if False, the returned den_graph
            contains only a single FSA
        Returns:
          A tuple (num_graph, den_graph), where

            - `num_graph` is the numerator graph. It is an FsaVec with
              shape `(len(texts), None, None)`.

            - `den_graph` is the denominator graph. It is an FsaVec
              with the same shape of the `num_graph` if replicate_den is
              True; otherwise, it is an FsaVec containing only a single FSA.
        """
        transcript_fsa = self.convert_transcript_to_fsa(texts)

        # remove word IDs from transcript_fsa since it is not needed
        del transcript_fsa.aux_labels
        # NOTE: You can comment out the above statement
        # if you want to run test/test_mmi_graph_compiler.py

        transcript_fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(
            transcript_fsa
        )

        transcript_fsa_with_self_loops = k2.arc_sort(transcript_fsa_with_self_loops)

        num = k2.compose(
            self.ctc_topo_P,
            transcript_fsa_with_self_loops,
            treat_epsilons_specially=False,
        )

        # CAUTION: Due to the presence of P,
        # the resulting `num` may not be connected
        num = k2.connect(num)

        num = k2.arc_sort(num)

        ctc_topo_P_vec = k2.create_fsa_vec([self.ctc_topo_P])
        if replicate_den:
            indexes = torch.zeros(len(texts), dtype=torch.int32, device=self.device)
            den = k2.index_fsa(ctc_topo_P_vec, indexes)
        else:
            den = ctc_topo_P_vec

        return num, den

    # def convert_transcript_to_fsa(self, texts: List[str]) -> k2.Fsa:
    #     """Convert transcripts to an FsaVec with the help of a lexicon
    #     and word symbol table.

    #     Args:
    #       texts:
    #         Each element is a transcript containing words separated by space(s).
    #         For instance, it may be 'HELLO icefall', which contains
    #         two words.

    #     Returns:
    #       Return an FST (FsaVec) corresponding to the transcript.
    #       Its `labels` is token IDs and `aux_labels` is word IDs.
    #     """
    #     word_ids_list = []
    #     for text in texts:
    #         word_ids = []
    #         for word in text.split():
    #             if word in self.lexicon.word_table:
    #                 word_ids.append(self.lexicon.word_table[word])
    #             else:
    #                 word_ids.append(self.oov_id)
    #         word_ids_list.append(word_ids)

    #     fsa = k2.linear_fsa(word_ids_list, self.device)
    #     fsa = k2.add_epsilon_self_loops(fsa)

    #     # The reason to use `invert_()` at the end is as follows:
    #     #
    #     # (1) The `labels` of L_inv is word IDs and `aux_labels` is token IDs
    #     # (2) `fsa.labels` is word IDs
    #     # (3) after intersection, the `labels` is still word IDs
    #     # (4) after `invert_()`, the `labels` is token IDs
    #     #     and `aux_labels` is word IDs
    #     transcript_fsa = k2.intersect(
    #         self.L_inv, fsa, treat_epsilons_specially=False
    #     ).invert_()
    #     transcript_fsa = k2.arc_sort(transcript_fsa)
    #     return transcript_fsa

    # def get_transcript_tokens(self, texts: List[str], out_file: str):
    #     """Convert a list of texts to a list-of-list of piece IDs.
    #     In our case, piece IDs are simply character units (in str form).
    #     The resulted list of tokens (for each utterance) will be saved
    #     in (presumably) the lang directory.

    #     Args:
    #       texts:
    #         It is a list of strings. Each string consists of space(s)
    #         separated words. An example containing two strings is given below:

    #             ['HELLO ICEFALL', 'HELLO k2']
    #         We assume it contains no OOVs. Otherwise, it will raise an
    #         exception.
    #       out_file:
    #         The file to save the tokens in. For the example above, the
    #         contents of out_file will be:
    #             H E L L O <eow> I C E F A L L
    #             H E L L O <eow> k 2
    #     """
    #     transcript_tokens = []
    #     for text in texts:
    #         tokens = []
    #         for word in text.split():
    #             if word in self.word_table:
    #                 tokens.extend(" ".join(list(word)))
    #             else:
    #                 raise ValueError(f"OOV word {word} found in transcript")
    #         transcript_tokens.append(" <space> ".join(tokens))
    #     with open(out_file, "w") as f:
    #         f.write("\n".join(transcript_tokens))