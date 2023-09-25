
import os
import shutil
import logging
from pathlib import Path
from typing import List, Optional

from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.k2_integration.lexicon import Lexicon
from speechbrain.k2_integration.make_kn_lm import make_kn_lm
import sentencepiece as spm


logger = logging.getLogger(__name__)


def get_lexicon(
        lang_dir,
        csv_files,
        extra_vocab_files,
        add_word_boundary=True,
        unit_type="char",
        tokenizer: Optional[SentencePiece] = None,
    ):
    '''Read csv_files to generate a $lang_dir/lexicon.txt for k2 training.
    This usually includes the csv files of the training set and the dev set in the output_folder.
    During training, we need to make sure that the lexicon.txt contains all (or the majority of) 
    the words in the training set and the dev set.

    Args:
        lang_dir: str
            the directory to store the lexicon.txt
        csv_files: List[str]
            a list of csv file paths which contain a transcript at their last column.
        extra_vocab_files: List[str]
            a list of extra vocab files, librispeech-vocab.txt is an example
        add_word_boundary: bool
            whether to add word boundary symbols <eow> at the end of each line to the 
            lexicon for every word. Only used when unit_type="char".
        unit_type: str
            the type of the units used in the lexicon. Can be "char" or "bpe".
        tokenizer: spm.SentencePieceProcessor
            the tokenizer used to tokenize the words. Only used when unit_type="bpe".

    Note that in each csv_file, the first line is the header, and the remaining lines are in the following format:

    ID, duration, wav, spk_id, wrd (transcription)

    We only need the transcription in this function.

    Returns:
    None

    Writes out $lang_dir/lexicon.txt

    Note that the lexicon.txt is a text file with the following format:
    word1 phone1 phone2 phone3 ...
    word2 phone1 phone2 phone3 ...

    In this code, we simply use the characters in the word as the phones.
    You can use other phone sets, e.g., phonemes, BPEs, to train a better model.
    '''
    def tokenize(word, add_word_boundary=True):
        if unit_type == "char":
            if add_word_boundary:
                return list(word) + ["<eow>"]
            return list(word)
        elif unit_type == "bpe":
            # assert tokenizer is not None
            return tokenizer.sp.encode_as_pieces(word)
    # Read train.csv, dev-clean.csv to generate a lexicon.txt for k2 training
    lexicon = dict()
    for file in csv_files:
        with open(file) as f:
            # Omit the first line
            f.readline()
            # Read the remaining lines
            for line in f:
                # Split the line 
                try:
                    trans = line.strip().split(",")[-1]
                except ValueError as e:
                    print(line.strip().split(","))
                    raise e
                # Split the transcription into words
                words = trans.split()
                for word in words:
                    if word not in lexicon:
                        lexicon[word] = tokenize(word, add_word_boundary)

    for file in extra_vocab_files:
        with open(file) as f:
            for line in f:
                # Split the line 
                word = line.strip().split()[0]
                # Split the transcription into words
                if word not in lexicon:
                    lexicon[word] = tokenize(word, add_word_boundary)
    # Write the lexicon to lang_dir/lexicon.txt
    os.makedirs(lang_dir, exist_ok=True)
    with open(os.path.join(lang_dir, "lexicon.txt"), "w") as f:
        fc = "<UNK> <unk>\n"
        for word in lexicon:
            fc += word + " " + " ".join(lexicon[word]) + "\n"
        f.write(fc)

def arpa_to_fst(
        arpa_dir: Path,
        output_dir: Path,
        words_txt: Path,
        disambig_symbol: str = "#0",
        convert_4gram: bool = True,
        suffix: Optional[str] = "",
    ):
    """ Use kaldilm to convert an ARPA LM to FST. For example, in librispeech
    you can find a 3-gram (pruned) and a 4-gram ARPA LM in the openslr
    website (https://www.openslr.org/11/). You can use this function to
    convert them to FSTs. The resulting FSTs can then be used to create a
    decoding graph (HLG) for k2 decoding.

    If `convert_4gram` is True, then we will convert the 4-gram ARPA LM to
    FST. Otherwise, we will only convert the 3-gram ARPA LM to FST.
    It is worth noting that if the fsts already exist in the output_dir,
    then we will not convert them again (so you may need to delete them
    by hand if you, at any point, change your ARPA model).

    Args:
        arpa_dir: Path to the directory containing the ARPA LM (we expect
            a file named 3-gram.pruned.1e-7.arpa to exist, and if
            `convert_4gram` is True, then "4-gram.arpa" should also exist).
        output_dir: Path to the directory where the FSTs will be saved.
        words_txt: Path to the words.txt file created by prepare_lang.
        disambig_symbol: The disambiguation symbol to use.
        convert_4gram: If True, then we will convert the 4-gram ARPA LM to
            FST. Otherwise, we will only convert the 3-gram ARPA LM to FST.
        suffix: A suffix to add to the 3gram (and 4gram) FST. This is useful
            when you want to run multiple experiments that use different LMs
            (i.e. the words.txt file that's used for their creation is 
            different in each case).
    
    Raises:
        ImportError: If kaldilm is not installed.
    """
    assert arpa_dir.is_dir()
    assert output_dir.is_dir()
    try:
        from kaldilm.arpa2fst import arpa2fst
    except ImportError:
        # This error will occur when there is fst LM in the provided lm_dir
        # and we are trying to create it by converting an ARPA LM to FST.
        # For this, we need to install kaldilm.
        raise ImportError(
            "Optional dependencies must be installed to use kaldilm.\n"
            "Install using `pip install kaldilm`."
        )
    def _arpa_to_fst_single(arpa_path: Path, out_fst_path: Path, max_order: int):
        """Convert a single ARPA LM to FST."""
        if out_fst_path.exists():
            return
        if not arpa_path.exists():
            raise FileNotFoundError(f"{arpa_path} not found while trying to create the {max_order} FST.")
        try:
            logger.info("Converting {} to FST".format(arpa_path))
            s = arpa2fst(
                input_arpa=str(arpa_path),
                disambig_symbol=disambig_symbol,
                read_symbol_table=str(words_txt),
                max_order=max_order,
            )
        except Exception as e:
            logger.info(f"Failed to create {max_order}-gram FST from input={arpa_path}, disambig_symbol={disambig_symbol}, read_symbol_table={words_txt}")
            raise e
        logger.info(f"Writing {out_fst_path}")
        with open(out_fst_path, "w") as f:
            f.write(s)
    # 3-gram arpa to fst conversion...
    arpa_path = arpa_dir / "3-gram.pruned.1e-7.arpa"
    fst_path = output_dir / f"G_3_gram{suffix}.fst.txt"
    _arpa_to_fst_single(arpa_path, fst_path, max_order=3)
    # Optionnal 4-gram arpa to fst conversion
    if convert_4gram:
        # arpa_path = arpa_dir / "4-gram.arpa"
        arpa_path = arpa_dir / "4-gram.arpa"
        fst_path = output_dir / f"G_4_gram{suffix}.fst.txt"
        _arpa_to_fst_single(arpa_path, fst_path, max_order=4)

def create_P_fst(
        lexicon: Lexicon,
        csv_path: str,
        output_dir: str,
        tokens_txt: str,
        disambig_symbol: str = "#0",
        max_order: int = 2,
        model_name: str = "P"
    ):
    """Create the P.fst.txt for LF-MMI. The reason we don't use `arpa_to_fst`
    is because this is a token-level LM (e.g. phone/word-piece/character LM).
    TODO: This could still be merged in `arpa_to_fst` by adding an extra
          `read_symbol_table` argument.

    Args:
        lexicon: The lexicon object used to get the tokenized version of 
            each transcript
        csv_path: The path to the csv file.
        output_dir: The directory where the P.arpa will be saved.
        tokens_txt: The path to the tokens.txt file created by prepare_lang.
        disambig_symbol: The disambiguation symbol to use.        
    """
    arpa_path = Path(output_dir) / f"{model_name}.arpa"
    fst_path = Path(output_dir) / f"{model_name}.fst.txt"
    if fst_path.exists():
        return
    if not arpa_path.exists():
        logger.info(f"Creating {arpa_path}")
        with open(csv_path) as f:
            texts = [line.strip().split(",")[-1] for line in f.readlines()[1:]]
        tokenized_transcripts = list(lexicon.generate_tokenized_transcripts(texts))
        tok_transcripts_path = Path(tokens_txt).parent / "tokenized_transcripts.txt"
        with open(tok_transcripts_path, "w") as f:
            logger.info(f"Writing {tok_transcripts_path}")
            f.write("\n".join(tokenized_transcripts))
        # Generate kneser-ney language model as arpa format. By default,
        # it will read the corpus from standard input, and output to 
        # standard output. Adapted from icefall's librispeech recipe.
        make_kn_lm(
            text=tok_transcripts_path,
            lm=arpa_path,
            ngram_order=max_order,
        )
    logger.info(f"Creating {fst_path}")
    try:
        from kaldilm.arpa2fst import arpa2fst
        s = arpa2fst(
            input_arpa=str(arpa_path),
            disambig_symbol=disambig_symbol,
            read_symbol_table=str(tokens_txt),
            max_order=max_order,
            bos_symbol="<s>",
            eos_symbol="</s>",
        )
    except Exception as e:
        logger.info(f"Failed to create 2-gram FST from input={arpa_path}, disambig_symbol={disambig_symbol}, read_symbol_table={tokens_txt}")
        raise e
    logger.info(f"Writing {fst_path}")
    with open(fst_path, "w") as f:
        f.write(s)



def get_bpe_tokenizer(hparams) -> spm.SentencePieceProcessor:
    """Get the BPE tokenizer. If the BPE model does not exist, then we will
    train it using SentencePiece.

    Args:
        hparams: The hyperparameters from a yaml file (e.g. hparams/train_hf_wav2vec_k2_mmi_bpe.yaml)

    Returns:
        The SentencePiece tokenizer.
    """
    n_tokens = hparams["output_neurons"]
    model_prefix = Path(hparams["lang_dir"]) / f"bpe_{n_tokens}"
    model_file = model_prefix.with_suffix(".model")
    if not model_file.is_file():
        with open(hparams["train_csv"]) as f:
            texts = [line.strip().split(",")[-1] for line in f.readlines()[1:]]
        transcripts_path = model_file.parent / "train_transcripts.txt"
        with open(transcripts_path, "w") as f:
            f.write("\n".join(texts))
        user_defined_symbols = ["<blk>", "<sos/eos>"]
        # if hparams["add_word_boundary"]:
        #     user_defined_symbols += ["<eow>"]
        unk_id = len(user_defined_symbols)
        logger.info(f"Saving a BPE model into {model_file}")
        spm.SentencePieceTrainer.train(
            input=str(transcripts_path),
            vocab_size=n_tokens,
            model_type="bpe",
            model_prefix=f"bpe_{n_tokens}",
            input_sentence_size=100000000,
            character_coverage=1.0,
            user_defined_symbols=user_defined_symbols,
            unk_id=unk_id,
            bos_id=-1,
            eos_id=-1,
        )
        shutil.move(
            f"bpe_{n_tokens}.model",
            str(model_file),
        )
        shutil.move(
            f"bpe_{n_tokens}.vocab",
            str(model_prefix.with_suffix(".vocab")),
        )
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(str(model_file))
    return tokenizer