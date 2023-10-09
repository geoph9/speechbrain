"""Graph compiler class to create, store, and use k2 decoding graphs in
speechbrain. The addition of a decoding graph, be it HL or HLG (with LM),
limits the output words to the ones in the lexicon. On top of that, a
bigger LM can be used to rescore the decoding graph and get better results.

This code is an extension of icefall's (https://github.com/k2-fsa/icefall)
graph compiler.

Authors:
  * Zeyu Zhao 2023
  * Georgios Karakasidis 2023
"""


import os
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Optional, Tuple, Iterable

import k2
import torch
import speechbrain as sb

from speechbrain.k2_integration.lexicon import Lexicon
from speechbrain.k2_integration.utils import get_texts, one_best_decoding, rescore_with_whole_lattice, get_lattice_or_prune


logger = logging.getLogger(__name__)


class GraphCompiler(ABC):
    def __init__(
        self,
        lexicon: Lexicon,
        device: torch.device,
        oov: str = "<UNK>",
        G_path: str = None,
        rescoring_lm_path: Union[Path, str] = None,
        decoding_method: str = "1best",
        subsampling_factor: int = 1,
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
          G_path: str
            Path to the language model FST to be used in the decoding-graph creation.
            If None, then we assume that the language model is not used.
          rescoring_lm_path: Path | str
            Path to the language model FST to be used in the rescoring of the decoding
            graph. If None, then we assume that the language model is not used.
          decoding_method: str
            One of 1best, whole-lattice-rescoring, or nbest.
          subsampling_factor: int
            Subsampling factor of the model. If > 1, then we need to subsample the
            decoding graph to match the output of the model.
        """
        # L_inv = lexicon.L_inv.to(device)
        # L = lexicon.L.to(device)
        self.lexicon = lexicon
        self.subsampling_factor = subsampling_factor
        assert self.lexicon.L_inv.requires_grad is False

        assert oov in lexicon.word_table

        # self.L_inv = k2.arc_sort(L_inv)
        # self.L = k2.arc_sort(L)
        self.oov_id = lexicon.word_table[oov]
        self.word_table = lexicon.word_table

        max_token_id = max(lexicon.tokens)
        ctc_topo = k2.ctc_topo(max_token_id, modified=False)
        self.ctc_topo = ctc_topo.to(device)
        self._L_inv = None

        self.device = device
        self.G_path: str = G_path
        self.rescoring_lm_path: Path = rescoring_lm_path
        self.decoding_graph: k2.Fsa = None  # HL or HLG
        self.rescoring_graph: k2.Fsa = None  # G (usually 4-gram LM)
        self.decoding_method: bool = decoding_method

    @property
    def L_inv(self) -> k2.Fsa:
        """Return the inverse of L (L_inv) as an FSA."""
        if self._L_inv is None:
            self._L_inv = k2.arc_sort(self.lexicon.L_inv).to(self.device)
        return self._L_inv

    def get_G(
            self,
            path: str = None,
            save: bool = True,
            device: Optional[torch.device] = None
        ) -> k2.Fsa:
        """Load a LM to be used in the decoding graph creation (or LM rescoring).
        Note that it doesn't load G into memory.

        Args:
            path: str, The path to an FST LM (ending with .fst.txt) or a k2-converted
                LM (in pytorch .pt format).
            save: bool, Whether or not to save the LM in .pt format (in the same dir).
        
        Returns:
            An FSA representing the LM. The device is the same as graph_compiler.device.
        """
        path = str(path or self.G_path)
        device = device or self.device
        if os.path.exists(path.replace(".fst.txt", ".pt")):
            logger.info(f"NOTE: Loading {path} from its .pt format")
            path = path.replace(".fst.txt", ".pt")
        # If G_path is an fst.txt file then convert to .pt file
        if path.endswith(".fst.txt"):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File {path} not found. You need to run the kaldilm to get it.")
            with open(path) as f:
                G = k2.Fsa.from_openfst(f.read(), acceptor=False).to(device)
        elif path.endswith(".pt"):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"File {path} not found.")
            d = torch.load(path, map_location=device)
            G = k2.Fsa.from_dict(d)#.to(self.device)
        else:
            raise ValueError(f"File {path} is not a .fst.txt or .pt file.")
        if save:
            torch.save(G.as_dict(), path[:-8] + ".pt")
        return G

    def get_rescoring_LM(self, path: str = None) -> k2.Fsa:
        """Load a LM with the purpose of using it for LM rescoring.
        For instance, in the librispeech recipe this is a 4-gram LM (while  a
        3gram LM is used for HLG construction).

        Args:
            path: str, The path to an FST LM (ending with .fst.txt) or a k2-converted
                LM (in pytorch .pt format).

        Returns:
            An FSA representing the LM. The device is the same as graph_compiler.device.
        """
        path = str(path or self.rescoring_lm_path)
        logger.info(f"Loading rescoring LM: {path}")
        G = self.get_G(path, save=False, device=torch.device("cpu"))
        del G.aux_labels
        G.labels[G.labels >= self.lexicon.word_table["#0"]] = 0
        G.__dict__["_properties"] = None
        G = k2.Fsa.from_fsas([G]).to('cpu')  # only used for decoding which is done in cpu
        G = k2.arc_sort(G)
        G = k2.add_epsilon_self_loops(G)
        G = k2.arc_sort(G)
        G = G.to(self.device)
        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        if not hasattr(G, "lm_scores"):
            G.lm_scores = G.scores.clone()
        return G

    def convert_transcript_to_fsa(self, texts: List[str]) -> k2.Fsa:
        """Convert a list of transcript texts to an FsaVec.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          Return an FsaVec, whose `shape[0]` equals to `len(texts)`.
        """
        word_ids_list = []
        for text in texts:
            word_ids = []
            for word in text.split():
                if word in self.word_table:
                    word_ids.append(self.word_table[word])
                else:
                    word_ids.append(self.oov_id)
            word_ids_list.append(word_ids)

        fsa = k2.linear_fsa(word_ids_list, self.device)
        fsa = k2.add_epsilon_self_loops(fsa)

        # The reason to use `invert_()` at the end is as follows:
        #
        # (1) The `labels` of L_inv is word IDs and `aux_labels` is token IDs
        # (2) `fsa.labels` is word IDs
        # (3) after intersection, the `labels` is still word IDs
        # (4) after `invert_()`, the `labels` is token IDs
        #     and `aux_labels` is word IDs
        transcript_fsa = k2.intersect(
            self.L_inv, fsa, treat_epsilons_specially=False
        ).invert_()
        transcript_fsa = k2.arc_sort(transcript_fsa)
        return transcript_fsa

        # word_fsa = k2.linear_fsa(word_ids_list, self.device)

        # word_fsa_with_self_loops = k2.add_epsilon_self_loops(word_fsa)

        # fsa = k2.intersect(
        #     self.L_inv, word_fsa_with_self_loops, treat_epsilons_specially=False
        # )
        # # fsa has word ID as labels and token ID as aux_labels, so
        # # we need to invert it
        # ans_fsa = fsa.invert_()
        # return k2.arc_sort(ans_fsa)
    
    @abstractmethod
    def compile(
            self,
            texts: List[str],
            *args, **kwargs
        ) -> Union[k2.Fsa, Tuple[k2.Fsa, k2.Fsa]]:
        """Build decoding graphs by composing ctc_topo with
        given transcripts.
        
        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
        
        Returns:
          An FsaVec, the composition result of `self.ctc_topo` and the
          transcript FSA.
          With MMI training, we need to return two FsaVecs, one for the numerator
          and one for the denominator.
        """
        pass

    def compile_HL(self):
        '''
        Compile the decoding graph by composing ctc_topo with L.
        This is for decoding without language model.
        Usually, you don't need to call this function explicitly.
        '''
        H = self.ctc_topo.to("cpu")
        logger.info("Arc sorting L")
        L = k2.arc_sort(self.lexicon.L).to("cpu")
        logger.info("Composing H and L")
        HL = k2.compose(H, L, inner_labels="tokens")

        logger.info("Connecting HL")
        HL = k2.connect(HL)

        logger.info("Arc sorting HL")
        self.decoding_graph = k2.arc_sort(HL)
        # self.decoding_graph = k2.arc_sort(H)
        torch.save(self.decoding_graph.as_dict(), self.lexicon.lang_dir / "HL.pt")

        logger.info(f"Number of arcs in the final HL: {HL.arcs.num_elements()}")

    def compile_HLG(self):
        '''
        Compile the decoding graph by composing ctc_topo with LG.
        This is for decoding with language model (by default we assume a 3gram lm).
        Usually, you don't need to call this function explicitly.
        '''
        H = self.ctc_topo.to("cpu")
        G = self.get_G(device=torch.device("cpu"))
        L = self.lexicon.L_disambig.to("cpu")

        first_token_disambig_id = self.lexicon.token_table["#0"]
        first_word_disambig_id = self.lexicon.word_table["#0"]
        
        L = k2.arc_sort(L)
        G = k2.arc_sort(G)
        logger.debug("Intersecting L and G")
        LG = k2.compose(L, G)

        logger.debug("Connecting LG")
        LG = k2.connect(LG)

        logger.debug("Determinizing LG")
        LG = k2.determinize(LG)

        logger.debug("Connecting LG after k2.determinize")
        LG = k2.connect(LG)

        logger.debug("Removing disambiguation symbols on LG")
        # NOTE: We need to clone here since LG.labels is just a reference to a tensor
        #       and we will end up having issues with misversioned updates on fsa's properties.
        labels = LG.labels.clone()
        labels[labels >= first_token_disambig_id] = 0
        LG.labels = labels

        assert isinstance(LG.aux_labels, k2.RaggedTensor)
        LG.aux_labels.values[LG.aux_labels.values >= first_word_disambig_id] = 0

        LG = k2.remove_epsilon(LG)

        LG = k2.connect(LG)
        LG.aux_labels = LG.aux_labels.remove_values_eq(0)
        logger.debug("Arc sorting LG")
        LG = k2.arc_sort(LG)

        logger.debug("Composing H and LG")
        HLG = k2.compose(H, LG, inner_labels="tokens")

        logger.debug("Connecting HLG")
        HLG = k2.connect(HLG)

        logger.debug("Arc sorting HLG")
        HLG = k2.arc_sort(HLG)

        logger.info(f"Number of arcs in the final HLG: {HLG.arcs.num_elements()}")

        self.decoding_graph = HLG


class CtcTrainingGraphCompiler(GraphCompiler):
    def __init__(
        self,
        lexicon: Lexicon,
        device: torch.device,
        oov: str = "<UNK>",
        need_repeat_flag: bool = False,
        G_path: str = None,
        rescoring_lm_path: Union[Path, str] = None,
        decoding_method: str = "1best",
        subsampling_factor: int = 1,
    ):
        """
        Args:
          lexicon: Lexicon
            It is built from `data/lang/lexicon.txt`.
          device: torch.device | str
            The device to use for operations compiling transcripts to FSAs.
          oov: str
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
          rescoring_lm_path: Path | str
            Path to the language model FST to be used in the rescoring of the decoding
            graph. If None, then we assume that the language model is not used.
          decoding_method: str
            One of 1best, whole-lattice-rescoring, or nbest.
          subsampling_factor: int
            Subsampling factor of the model. If > 1, then we need to subsample the
            decoding graph to match the output of the model.
        """
        
        super().__init__(
            lexicon=lexicon,
            device=device,
            oov=oov,
            G_path=G_path,
            rescoring_lm_path=rescoring_lm_path,
            decoding_method=decoding_method,
            subsampling_factor=subsampling_factor,
        )

        if need_repeat_flag:
            self.ctc_topo._is_repeat_token_ = (
                self.ctc_topo.labels != self.ctc_topo.aux_labels
            )

    def compile(self, texts: List[str]) -> k2.Fsa:
        """Build decoding graphs by composing ctc_topo with
        given transcripts.

        Args:
          texts:
            A list of strings. Each string contains a sentence for an utterance.
            A sentence consists of spaces separated words. An example `texts`
            looks like:

                ['hello icefall', 'CTC training with k2']

        Returns:
          An FsaVec, the composition result of `self.ctc_topo` and the
          transcript FSA.
        """
        transcript_fsa = self.convert_transcript_to_fsa(texts)

        # NOTE: k2.compose runs on CUDA only when treat_epsilons_specially
        # is False, so we add epsilon self-loops here
        fsa_with_self_loops = k2.remove_epsilon_and_add_self_loops(
            transcript_fsa)

        fsa_with_self_loops = k2.arc_sort(fsa_with_self_loops)

        decoding_graph = k2.compose(
            self.ctc_topo, fsa_with_self_loops, treat_epsilons_specially=False
        )

        assert decoding_graph.requires_grad is False

        return decoding_graph

    def decode(self,
               log_probs: torch.Tensor,
               input_lens: torch.Tensor,
               search_beam=5,
               output_beam=5,
               ac_scale=1.0,
               min_active_states=300,
               max_active_states=1000,
               is_test: bool = True,
               stage: sb.Stage = sb.Stage.TEST,
               lm_scale_list: Optional[List[float]] = None,
               rescoring_lm_path: Optional[Path] = None,
               force_device: Optional[torch.device] = None,
        ) -> Union[List[str], Dict[str, List[str]]]:
        """
        Decode the given log_probs with self.decoding_graph without language model.

        Args:
          log_probs:
            It is an input tensor of shape (batch, seq_len, num_tokens).
          input_lens:
            It is an int tensor of shape (batch,). It contains lengths of
            each sequence in `log_probs`.
          search_beam: int, decoding beam size
          output_beam: int, lattice beam size
          ac_scale: float, acoustic scale applied to `log_probs`
          min_active_states: int, minimum #states that are not pruned during decoding
          max_active_states: int, maximum #active states that are kept during decoding
          is_test: bool, if testing is performed then we won't log warning about <UNK>s.
          stage: sb.Stage, the stage of the experiment (Usually VALID or TEST).
          lm_scale_list: List[float], a list of language model scale factors. Defaults to [0.6].
          rescoring_lm_path: Path, path to the LM to be used for rescoring. If not provided
            and the decoding method is whole-lattice-rescoring, then you need to provide
            the `rescoring_lm_path` in the constructor of this class.
          force_device: torch.device, if provided, then the decoding graph will be moved
            to this device before decoding.

        Returns:
          If decoding_method==1best: a list of strings, each of which is the decoding 
            result of the corresponding utterance.
          If decoding_method==whole-lattice-rescoring: a dict of lists of strings, each of
            which is the decoding result of the corresponding utterance. The keys of the dict
            are the language model scale factors used for rescoring.
        """
        lm_scale_list = lm_scale_list or [0.6]
        # force_device = torch.device("cpu")
        if force_device and isinstance(force_device, torch.device):
            device = force_device
            log_probs = log_probs.to(device)
        else:
            device = log_probs.device
        if self.decoding_graph is None:
            # Disable logging of unknown words if we are in test stage
            if is_test:
                self.lexicon.log_unknown_warning = False
            if self.G_path is None:
                self.compile_HL()
            else:
                logger.info("Compiling HLG instead of HL")
                self.compile_HLG()
                # if not hasattr(self.decoding_graph, "lm_scores"):
                #     self.decoding_graph.lm_scores = self.decoding_graph.scores.clone()
            # Delete lexicon's L and L_inv to save memory
            if hasattr(self.lexicon, "L"):
                del self.lexicon.L
            self._L_inv = None
            if self.decoding_graph.device != device:
                self.decoding_graph = self.decoding_graph.to(device)
            if self.decoding_method == "whole-lattice-rescoring":
                # fst_4gram_path = str(Path(self.G_path).parent / "G_4_gram.fst.txt")
                # fst_4gram_path = "lm/G_4_gram_withfullwords.fst.txt"
                self.rescoring_graph = self.get_rescoring_LM(rescoring_lm_path).to(device)
        # If stage != TEST then we always do 1best decoding
        decoding_method = self.decoding_method if stage == sb.Stage.TEST else "1best"
        
        input_lens = input_lens.to(device)

        input_lens = (input_lens * log_probs.shape[1]).round().int()
        # NOTE: low ac_scales may results in very big lattices and OOM errors.
        log_probs *= ac_scale

        def lattice2text(best_path: k2.Fsa) -> List[str]:
            """Convert the best path to a list of strings."""
            hyps: List[List[int]] = get_texts(best_path, return_ragged=False)
            texts = []
            for wids in hyps:
                texts.append(" ".join([self.word_table[wid]
                            for wid in wids if wid != self.oov_id]))
            return texts

        with torch.no_grad():
            torch.cuda.empty_cache()
            lattice = get_lattice_or_prune(
                log_probs,
                input_lens,
                self,
                search_beam=search_beam,
                output_beam=output_beam,
                min_active_states=min_active_states,
                max_active_states=max_active_states,
            )
            if decoding_method == "1best":
                key = "no_rescore"
                best_path = {
                    key: one_best_decoding(
                        lattice=lattice, use_double_scores=True
                    )
                }
                out = lattice2text(best_path[key])
            elif decoding_method == "whole-lattice-rescoring":
                best_path = rescore_with_whole_lattice(
                    lattice=lattice.to(device),
                    G_with_epsilon_loops=self.rescoring_graph,
                    lm_scale_list=lm_scale_list,
                    use_double_scores=True,
                )
                out = {}
                for lm_scale in lm_scale_list:
                    key = f"lm_scale_{lm_scale:.1f}"
                    out[key] = lattice2text(best_path[key])
            else:
                raise ValueError(f"Decoding method '{decoding_method}' is not supported.")
            del lattice
            del best_path
            torch.cuda.empty_cache()

            return out

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
        decoding_method: str = "1best",
        subsampling_factor: int = 1,
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
          decoding_method: str
            One of 1best, whole-lattice-rescoring, or nbest.
          subsampling_factor: int
            Subsampling factor of the model. If > 1, then we need to subsample the
            decoding graph to match the output of the model.
        """
        super().__init__(
            lexicon=lexicon,
            device=device,
            oov=oov,
            need_repeat_flag=need_repeat_flag,
            G_path=G_path,
            rescoring_lm_path=rescoring_lm_path,
            decoding_method=decoding_method,
            subsampling_factor=subsampling_factor,
        )
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.P_path = P_path
        self._p = None
        self._ctc_topo_P = None

    @property
    def ctc_topo_P(self):
        if self._ctc_topo_P is None:
            # build it on demand
            self.build_ctc_topo_P()
        return self._ctc_topo_P.to(self.device)

    @property
    def P(self) -> k2.Fsa:
        if isinstance(self._p, k2.Fsa):
            return self._p
        with open(self.P_path) as f:
            # P is not an acceptor because there is
            # a back-off state, whose incoming arcs
            # have label #0 and aux_label 0 (i.e., <eps>).
            self._p = k2.Fsa.from_openfst(f.read(), acceptor=False)
        return self._p
    
    @staticmethod
    def create_flower_fst(table_file):
        # Use for debugging
        with open(table_file, "r") as f:
            mapping = {w.split()[0]: w.split()[1] for w in f.readlines()}
        fst = "\n"
        for value in mapping.values():
            fst += f"0 0 {value} {value}\n"
        fst += "0"
        return fst

    def build_ctc_topo_P(self):
        """Build ctc_topo_P, the composition result of
        ctc_topo and P, where P is a pre-trained n-gram
        word piece LM.
        """
        # Note: there is no need to save a pre-compiled P and ctc_topo
        # as it is very fast to generate them.
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
        logger.info(
            f"Building ctc_topo (modified=False). max_token_id: {max_token_id}"
        )

        ctc_topo_inv = k2.arc_sort(self.ctc_topo.invert())

        ctc_topo_P = k2.intersect(
            ctc_topo_inv, P_with_self_loops, treat_epsilons_specially=False
        ).invert()

        self._ctc_topo_P = k2.arc_sort(ctc_topo_P)
        logger.info(f"ctc_topo_P num_arcs: {self.ctc_topo_P.num_arcs}")

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
    
    def decode(self,
               log_probs: torch.Tensor,
               input_lens: torch.Tensor,
               search_beam=5,
               output_beam=5,
               ac_scale=1.0,
               min_active_states=300,
               max_active_states=1000,
               is_test: bool = True,
               stage: sb.Stage = sb.Stage.TEST,
               lm_scale_list: Optional[List[float]] = None,
               rescoring_lm_path: Optional[Path] = None,
               force_device: Optional[torch.device] = None,
        ) -> Union[List[str], Dict[str, List[str]]]:
        # Make sure we cleanup unsused variables for decoding
        # if getattr(self, "_p", None):
        #     # delete all unnecessary attributes to save memory
        #     del self._ctc_topo_P
        #     self._ctc_topo_P = None
        #     self._p = None
        return super().decode(
            log_probs=log_probs,
            input_lens=input_lens,
            search_beam=search_beam,
            output_beam=output_beam,
            ac_scale=ac_scale,
            min_active_states=min_active_states,
            max_active_states=max_active_states,
            is_test=is_test,
            stage=stage,
            lm_scale_list=lm_scale_list,
            rescoring_lm_path=rescoring_lm_path,
            force_device=force_device,
        )