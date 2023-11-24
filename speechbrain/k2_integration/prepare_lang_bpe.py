#!/usr/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
# Modified for integration with SpeechBrain 2023 the University of Edinburgh (Zeyu Zhao, Georgios Karakasidis)


import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import k2
import torch
import sentencepiece as spm

from speechbrain.k2_integration.lexicon import read_lexicon, write_lexicon
from speechbrain.k2_integration.prepare_lang import add_disambig_symbols, lexicon_to_fst

Lexicon = List[Tuple[str, List[str]]]


def write_mapping(filename: str, sym2id: Dict[str, int]) -> None:  #TODO
    """Write a symbol to ID mapping to a file.

    Note:
      No need to implement `read_mapping` as it can be done
      through :func:`k2.SymbolTable.from_file`.

    Args:
      filename:
        Filename to save the mapping.
      sym2id:
        A dict mapping symbols to IDs.
    Returns:
      Return None.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for sym, i in sym2id.items():
            f.write(f"{sym} {i}\n")

# #TODO
# def get_tokens(lexicon: Lexicon) -> List[str]:
#     """Get tokens from a lexicon.

#     Args:
#       lexicon:
#         It is the return value of :func:`read_lexicon`.
#     Returns:
#       Return a list of unique tokens.
#     """
#     ans = set()
#     for _, tokens in lexicon:
#         ans.update(tokens)
#     sorted_ans = sorted(list(ans))
#     return sorted_ans
def get_tokens(lexicon: Lexicon, sil_token="SIL", manually_add_sil_to_tokens=False, unk_token="<unk>") -> List[str]:
    """Get tokens from a lexicon.

    Args:
      lexicon:
        It is the return value of :func:`read_lexicon`.
      sil_token:
        The optional silence token between words. It should not appear in the lexicon, otherwise it will cause an error.
    Returns:
      Return a list of unique tokens.
    """
    ans = set()
    for _, tokens in lexicon:
        # assert sil_token not in tokens, f"{sil_token} should not appear in the lexicon but it is found in {_}"
        ans.update(tokens)
    # remove unk_token and add it later (it's id needs to be after sil but before any other token)
    ans.remove(unk_token)
    if manually_add_sil_to_tokens:
        ans.remove(sil_token)
    sorted_ans = sorted(list(ans))
    sorted_ans = [unk_token] + sorted_ans
    if manually_add_sil_to_tokens:
        sorted_ans = [sil_token] + sorted_ans
    return sorted_ans

#TODO
def get_words(lexicon: Lexicon) -> List[str]:
    """Get words from a lexicon.

    Args:
      lexicon:
        It is the return value of :func:`read_lexicon`.
    Returns:
      Return a list of unique words.
    """
    ans = set()
    for word, _ in lexicon:
        ans.add(word)
    sorted_ans = sorted(list(ans))
    return sorted_ans

def generate_id_map(symbols: List[str]) -> Dict[str, int]:
    """Generate ID maps, i.e., map a symbol to a unique ID.

    Args:
      symbols:
        A list of unique symbols.
    Returns:
      A dict containing the mapping between symbols and IDs.
    """
    return {sym: i for i, sym in enumerate(symbols)}

#TODO
def add_self_loops(
    arcs: List[List[Any]], disambig_token: int, disambig_word: int
) -> List[List[Any]]:
    """Adds self-loops to states of an FST to propagate disambiguation symbols
    through it. They are added on each state with non-epsilon output symbols
    on at least one arc out of the state.

    See also fstaddselfloops.pl from Kaldi. One difference is that
    Kaldi uses OpenFst style FSTs and it has multiple final states.
    This function uses k2 style FSTs and it does not need to add self-loops
    to the final state.

    The input label of a self-loop is `disambig_token`, while the output
    label is `disambig_word`.

    Args:
      arcs:
        A list-of-list. The sublist contains
        `[src_state, dest_state, label, aux_label, score]`
      disambig_token:
        It is the token ID of the symbol `#0`.
      disambig_word:
        It is the word ID of the symbol `#0`.

    Return:
      Return new `arcs` containing self-loops.
    """
    states_needs_self_loops = set()
    for arc in arcs:
        src, dst, ilabel, olabel, score = arc
        if olabel != 0:
            states_needs_self_loops.add(src)

    ans = []
    for s in states_needs_self_loops:
        ans.append([s, s, disambig_token, disambig_word, 0])

    return arcs + ans


def lexicon_to_fst_no_sil(
    lexicon: Lexicon,
    token2id: Dict[str, int],
    word2id: Dict[str, int],
    need_self_loops: bool = False,
) -> k2.Fsa:
    """Convert a lexicon to an FST (in k2 format).

    Args:
      lexicon:
        The input lexicon. See also :func:`read_lexicon`
      token2id:
        A dict mapping tokens to IDs.
      word2id:
        A dict mapping words to IDs.
      need_self_loops:
        If True, add self-loop to states with non-epsilon output symbols
        on at least one arc out of the state. The input label for this
        self loop is `token2id["#0"]` and the output label is `word2id["#0"]`.
    Returns:
      Return an instance of `k2.Fsa` representing the given lexicon.
    """
    loop_state = 0  # words enter and leave from here
    next_state = 1  # the next un-allocated state, will be incremented as we go

    arcs = []

    # assert token2id["<blk>"] == 0
    assert word2id["<eps>"] == 0

    eps = 0

    for word, pieces in lexicon:
        assert len(pieces) > 0, f"{word} has no pronunciations"
        cur_state = loop_state

        word = word2id[word]
        pieces = [token2id[i] for i in pieces]

        for i in range(len(pieces) - 1):
            w = word if i == 0 else eps
            arcs.append([cur_state, next_state, pieces[i], w, 0])

            cur_state = next_state
            next_state += 1

        # now for the last piece of this word
        i = len(pieces) - 1
        w = word if i == 0 else eps
        arcs.append([cur_state, loop_state, pieces[i], w, 0])

    if need_self_loops:
        disambig_token = token2id["#0"]
        disambig_word = word2id["#0"]
        arcs = add_self_loops(
            arcs,
            disambig_token=disambig_token,
            disambig_word=disambig_word,
        )

    final_state = next_state
    arcs.append([loop_state, final_state, -1, -1, 0])
    arcs.append([final_state])

    arcs = sorted(arcs, key=lambda arc: arc[0])
    arcs = [[str(i) for i in arc] for arc in arcs]
    arcs = [" ".join(arc) for arc in arcs]
    arcs = "\n".join(arcs)

    fsa = k2.Fsa.from_str(arcs, acceptor=False)
    return fsa


def prepare_lang(
        lang_dir,
        sil_token="<sil>",
        sil_prob=0.5,
        unk_token="<unk>",
    ):
    """
    This function takes as input a lexicon file "$lang_dir/lexicon.txt"
    consisting of words and tokens (i.e., phones) and does the following:

    1. Add disambiguation symbols to the lexicon and generate lexicon_disambig.txt

    2. Generate tokens.txt, the token table mapping a token to a unique integer.

    3. Generate words.txt, the word table mapping a word to a unique integer.

    4. Generate L.pt, in k2 format. It can be loaded by

            d = torch.load("L.pt")
            lexicon = k2.Fsa.from_dict(d)

    5. Generate L_disambig.pt, in k2 format.


    Args:

    lang_dir: The directory to store the output files and read the input file lexicon.txt.
    sil_token: The silence token.
    sil_prob: The probability of adding silence to the tokens. If 0, then silence is not added to the tokens.
    unk_token: The unknown token. Default is <unk>.
    """

    out_dir = Path(lang_dir)
    lexicon_filename = out_dir / "lexicon.txt"

    # backup the original L.pt, L_disambig.pt, tokens.txt and words.txt, Linv.pt and lexicon_disambig.txt
    for f in ["L.pt", "L_disambig.pt", "tokens.txt", "words.txt", "Linv.pt", "lexicon_disambig.txt"]:
        if (out_dir / f).exists():
            os.makedirs(out_dir / "backup", exist_ok=True)
            logging.info(f"Backing up {out_dir / f} to {out_dir}/backup/{f}")
            os.rename(out_dir / f, out_dir / "backup" / f)

    lexicon = read_lexicon(lexicon_filename)
    
    if sil_prob != 0:
        # add silence to the tokens
        tokens = get_tokens(lexicon, sil_token=sil_token,
                            manually_add_sil_to_tokens=True,
                            unk_token=unk_token)
    else:
        tokens = get_tokens(lexicon, manually_add_sil_to_tokens=False, unk_token=unk_token)
    words = get_words(lexicon)

    lexicon_disambig, max_disambig = add_disambig_symbols(lexicon)

    for i in range(max_disambig + 1):
        disambig = f"#{i}"
        assert disambig not in tokens
        tokens.append(f"#{i}")

    assert "<eps>" not in tokens
    tokens = ["<eps>"] + tokens
    token_sym_table: Dict[str, int] = generate_id_map(tokens)

    assert "<eps>" not in words
    assert "#0" not in words
    assert "<s>" not in words
    assert "</s>" not in words

    words = ["<eps>"] + words + ["#0", "<s>", "</s>"]

    word2id = generate_id_map(words)

    write_mapping(out_dir / "tokens.txt", token_sym_table)
    write_mapping(out_dir / "words.txt", word2id)
    write_lexicon(out_dir / "lexicon_disambig.txt", lexicon_disambig)

    if sil_prob > 0:
        L = lexicon_to_fst(
            lexicon,
            token2id=token_sym_table,
            word2id=word2id,
            sil_token=sil_token,
            sil_prob=sil_prob,
        )
    else:
        L = lexicon_to_fst_no_sil(
            lexicon,
            token2id=token_sym_table,
            word2id=word2id,
        )
    if sil_prob > 0:
        L_disambig = lexicon_to_fst(
            lexicon_disambig,
            token2id=token_sym_table,
            word2id=word2id,
            sil_token=sil_token,
            sil_prob=sil_prob,
            need_self_loops=True,
        )
    else:
        L_disambig = lexicon_to_fst_no_sil(
            lexicon_disambig,
            token2id=token_sym_table,
            word2id=word2id,
            need_self_loops=True,
        )
    torch.save(L.as_dict(), out_dir / "L.pt")
    torch.save(L_disambig.as_dict(), out_dir / "L_disambig.pt")

    logging.info("Converting L.pt to Linv.pt")
    L_inv = k2.arc_sort(L.invert())
    torch.save(L_inv.as_dict(), out_dir / "Linv.pt")
