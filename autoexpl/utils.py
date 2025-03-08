import json, time, logging, os, sys, re
from typing import List, Tuple, Dict, Union
import glob

logger = logging.getLogger("")


class LoggingTimer:
    def __init__(self, log_start_name: str, log_mid_name: str = None):
        self.startname = log_start_name
        self.midname = log_mid_name
        self.starttime = time.time()
        self.time = self.starttime
        logger.info(f"Start {self.startname} {self.midname}")

    def mid(self, new_log_mid_name: str):
        logger.info(
            f"\tDone {self.midname if self.midname is not None else self.startname} {time.time() - self.time:.3f} sec"
        )
        self.midname = new_log_mid_name
        self.time = time.time()
        logger.info(f"Start {self.midname}")

    def end(self):
        if self.midname is not None:
            logger.info(f"\tDone {self.midname} {time.time() - self.time:.3f} sec")
        logger.info(f"\tDone {self.startname} {time.time() - self.starttime:.3f} sec")
        del self


def remove_duplicate_tuples(lst):
    seen = set()
    result = []
    for tup in lst:
        if tup[:1] not in seen:
            result.append(tup)
            seen.add(tup[:1])
    return result


def get_token_char_indices(sent, toks):
    """GPT4 Generated
    Assumption: Always true "".join(toks) == "".join(sent.split())
    No other character inserted than space
    sent = "I ate apple."
    toks = ["I", "ate", "apple", "."]
    [(0, 0), (2, 4), (6, 10), (11, 11)]
    """
    assert "".join(toks) == "".join(sent.split())
    indices = []
    curr_idx = 0
    for tok in toks:
        idx = sent.find(tok, curr_idx)
        assert idx != -1, f"No {tok=} in {sent=}"
        indices.append((idx, idx + len(tok) - 1))
        curr_idx = idx + len(tok)
    return indices


# printing
class color:
    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


def get_unaligned_idx(
    align_words: List[tuple], src_tok: List[str], tgt_tok: List[str]
) -> Tuple[List[int], List[int]]:
    src_idx = []
    tgt_idx = []
    for i, j in align_words:
        src_idx.append(i)
        tgt_idx.append(j)
    src_aligned = list(set(src_idx))
    tgt_aligned = list(set(tgt_idx))
    src_all = [i for i in range(len(src_tok))]
    tgt_all = [i for i in range(len(tgt_tok))]
    src_unaligned = [x for x in src_all if x not in src_aligned]
    tgt_unaligned = [x for x in tgt_all if x not in tgt_aligned]

    return src_unaligned, tgt_unaligned


def print_unaligned_toks(src_unaligned_txt: List[str], tgt_unaligned_txt: List[str]):
    # src_unaligned_txt = [src_tok[x] for x in src_unaligned]
    # tgt_unaligned_txt = [tgt_tok[x] for x in tgt_unaligned]
    print("## src unaligned tokens")
    print(f"{color.BOLD}{color.BLUE}{src_unaligned_txt}{color.END}")
    print("## tgt unaligned tokens")
    print(f"{color.BOLD}{color.RED}{tgt_unaligned_txt}{color.END}")


def print_aligned(align_words: List[tuple], src_tok: List[str], tgt_tok: List[str]):
    for i, j in sorted(align_words):
        print(
            f"{color.BOLD}{color.BLUE}{src_tok[i]}{color.END}==={color.BOLD}{color.RED}{tgt_tok[j]}{color.END}"
        )


def add_unaligned_highlight(
    tgt_tok: List[str], tgt_unaligned_txt_filtered: List[str]
) -> str:
    line = ""
    for t in tgt_tok:
        if t in tgt_unaligned_txt_filtered:
            line += f"{color.BOLD}{color.RED}{t}{color.END} "
        else:
            line += f"{color.BOLD}{color.BLUE}{t}{color.END} "
    return line


def add_unaligned_highlight_ner_underline(
    tgt_tok: List[str], tgt_unaligned_txt_filtered_idx: List[int], lne: List[int]
) -> str:
    line = ""
    for i, t in enumerate(tgt_tok):
        if i in tgt_unaligned_txt_filtered_idx:
            line += f'{color.BOLD}{color.UNDERLINE if i in lne else ""}{color.RED}{t}{color.END} '
        else:
            line += f'{color.BOLD}{color.UNDERLINE if i in lne else ""}{color.BLUE}{t}{color.END} '
    return line


def add_ner_underline(src_tok: List[str], slne: List[int], color_name="GREEN") -> str:
    sline = ""
    for i, x in enumerate(src_tok):
        if i in slne:
            sline += f"{color.BOLD}{getattr(color, color_name)}{color.UNDERLINE}{x}{color.END} "
        else:
            sline += f"{color.BOLD}{getattr(color, color_name)}{x}{color.END} "
    return sline
def add_highlight_tokidx(src_tok: List[str], slne: List[int], color_name="GREEN") -> str:
    sline = ""
    for i, x in enumerate(src_tok):
        if i in slne:
            sline += f"{color.BOLD}{getattr(color, color_name)}{color.RED}{x}{color.END} "
        else:
            sline += f"{getattr(color, color_name)}{x}{color.END} "
    return sline

def add_ner_underline_char(para, lrange, color_name="GREEN") -> str:
    sline = ""
    p = 0
    for i, range in enumerate(lrange):
        s = range[0]
        e = range[1]
        assert p <= s
        sline += f"{color.BOLD}{getattr(color, color_name)}{para[p:s]}{color.END}"
        sline += f"{color.BOLD}{getattr(color, color_name)}{color.UNDERLINE}{para[s:e+1]}{color.END}"
        p = e + 1
    sline += f"{color.BOLD}{getattr(color, color_name)}{para[p:]}{color.END}"
    return sline


def convert_ltuple2strdash(ltuple: List[tuple]) -> str:
    ltuple.sort()
    line = None
    for x, y in ltuple:
        if line is None:
            line = f"{x}-{y}"
        else:
            line += f" {x}-{y}"
    return line


def convert_strdash2ltuple(strdash: str) -> List[tuple]:
    tuplelist = []
    for xy in strdash.strip().split():
        x, y = xy.split("-")
        tuplelist.append((int(x), int(y)))
    return tuplelist


def get_infix_lang(dataset_name, pair, lang, pretokname, ner_name) -> str:
    return f"{dataset_name}_{pair}_{lang}_{pretokname}_{ner_name}"


def get_ner_file_name(dataset_name, pair, lang, pretokname, ner_name) -> str:
    return f"nerdict.{get_infix_lang(dataset_name, pair, lang, pretokname, ner_name)}"


def get_wikiid_file_name(dataset_name, pair, lang, pretokname, ner_name) -> str:
    return f"wikiid.{get_infix_lang(dataset_name, pair, lang, pretokname, ner_name)}"


def get_mgenre_file_name(dataset_name, pair, lang, pretokname, ner_name) -> str:
    return f"mgenre_sentcache.{get_infix_lang(dataset_name, pair, lang, pretokname, ner_name)}"


def get_detect_exp_name(
    dataset_name,
    pair,
    pretokname,
    ner_name,
    aligner_names,  # List[str]
    vote_threshold,
    proximity_threshold,
) -> str:
    n = f'{dataset_name}_{pair}_{pretokname}_{ner_name}_{"-".join(aligner_names)}_vm{vote_threshold}pt{proximity_threshold}'
    return n


def tup_range_to_idx(tupleidxrange):
    slne = []
    for t in tupleidxrange:
        s = t[0]
        e = t[1]
        for i in range(s, e + 1):
            slne.append(i)
    return slne


def lidx_tok2char(list_tok_idx_start_end, map_tok2char):
    lcharange = []
    for tok_idx_start_end in list_tok_idx_start_end:
        sti = tok_idx_start_end[0]
        eti = tok_idx_start_end[1]
        s, _ = map_tok2char[sti]
        _, e = map_tok2char[eti]
        lcharange.append([s, e])
    return lcharange


def get_partial_presufix_candidates(
    dataset_name, pair, nationality, vote_threshold, proximity_threshold, **kwargs
):
    return (
        f"{nationality}.{dataset_name}_{pair}_",
        f"_vm{vote_threshold}pt{proximity_threshold}",
    )


def get_gsheet_name(
    dataset_name,
    pair,
    nationality,
    vote_threshold,
    proximity_threshold,
    start_idx,
    start_sent_id,
    end_idx,
    end_sent_id,
    npair_per_file,
    version,
) -> str:
    dataset_name = dataset_name.replace("wikimatrix", "wm")
    n = f"{dataset_name}_{pair}_{nationality}_vm{vote_threshold}pt{proximity_threshold}_n{npair_per_file}i{start_idx}d{start_sent_id}Ti{end_idx}d{end_sent_id}v{version}"
    return n


def get_vars_from_gsheet_name(spreadsheet_name):
    pattern = r"^(?P<dataset_name>[^_]+)_(?P<pair>[^_]+)_(?P<nationality>[^_]+)_vm(?P<vote_threshold>[^_]+)pt(?P<proximity_threshold>[^_]+)_n(?P<npair_per_file>[^_]+)i(?P<start_idx>[^_]+)d(?P<start_sent_id>[^_]+)Ti(?P<end_idx>[^_]+)d(?P<end_sent_id>[^_]+)v(?P<version>[^_]+)$"
    match = re.match(pattern, spreadsheet_name)
    if match:
        var_dict = match.groupdict()
        var_dict["dataset_name"] = var_dict["dataset_name"].replace("wm", "wikimatrix")
        return var_dict
    else:
        NotImplementedError


def get_ner_file_path_langs(
    data_dir, dataset_name, langs, pretokname, ner_name
) -> Dict[str, str]:
    filepaths = {}
    pair = "".join(langs)
    for lang in langs:
        filepaths[lang] = os.path.join(
            data_dir, get_ner_file_name(dataset_name, pair, lang, pretokname, ner_name)
        )
    return filepaths


def get_alignment_file_name(
    dataset_name, src_lang, tgt_lang, pretokname, alignname
) -> str:
    return f"align.{dataset_name}_{src_lang}{tgt_lang}_{pretokname}_{alignname}"


def matched_alignment_file_name(
    data_dir, dataset_name, src_lang, tgt_lang, pretokname, aligner_names
) -> Dict[str, str]:
    file_dict = {}
    candlist = []
    for aligner_name in aligner_names:
        file_match = (
            f"align.{dataset_name}_{src_lang}{tgt_lang}_{pretokname}_{aligner_name}"
        )
        matched_list = glob.glob(f"{os.path.join(data_dir, file_match)}*")
        assert len(matched_list)
        candlist += matched_list
    for filepath in candlist:
        filename = os.path.split(filepath)[1]
        minus_key = f"align.{dataset_name}_{src_lang}{tgt_lang}_{pretokname}_"
        key = filename[len(minus_key) :]
        file_dict[key] = filepath
    return file_dict


def proximity_filter(
    la: List[int], lb: List[int], threshold=3
) -> Tuple[List[int], List[int]]:
    new_la = []
    new_lb = set()
    for a in la:
        tmp = []
        for b in lb:
            if abs(a - b) < threshold:
                tmp.append(b)
        if len(tmp):
            new_la.append(a)
            new_lb.update(set(tmp))
    return new_la, list(new_lb)


def proximity_filter_entitychunk(
    la: List[Tuple[int, int, str]], lb: List[int], threshold=3
) -> Tuple[List[Tuple[int, int, str]], List[List[int]], List[bool]]:
    new_la = []
    new_lb = []
    # possible_explanandum : unaligned is not included and closely located with unalinged.
    l_unaligned_not_included = []
    for a in la:
        unaligned_not_included = True
        s, e, t = a
        tmp = []
        for b in lb:
            if (s <= b and b <= e) or min(abs(s - b), abs(e - b)) < threshold:
                tmp.append(b)
                if s <= b and b <= e:
                    unaligned_not_included = False
        if len(tmp):
            new_la.append(a)
            new_lb.append(tmp)
            l_unaligned_not_included.append(unaligned_not_included)
    return new_la, new_lb, l_unaligned_not_included


def check_flip_or_duplicates(tokenizations, lang="", orig_id=""):
    psi, pei = -2, -2
    for i, toks in enumerate(tokenizations):
        si = toks[0]
        ei = toks[1]
        if psi == si and pei == ei:
            print(f"{lang} {orig_id} duplicate ")
            return False
        elif pei + 1 == si:
            print(f"{lang} {orig_id} right after")
        elif pei == si:
            print(f"{lang} {orig_id} one overlap right after")
        elif ei == psi:
            print(f"{lang} {orig_id} one overlap before")
        elif ei < psi:
            print(f"{lang} {orig_id} before ")
            return False
        elif si <= psi and ei <= pei:
            print(f"{lang} {orig_id} overlap front ")
            return False
        elif psi <= si and ei <= pei:
            print(f"{lang} {orig_id} included ")
            return False
        elif si <= psi and pei <= ei:
            print(f"{lang} {orig_id} inclusing ")
            return False
        elif si <= pei and pei <= ei:
            print(f"{lang} {orig_id} overlap back ")
            return False
        elif pei < si:
            # ideal
            aa = 0
        psi, pei = si, ei
    return True


def nationality_keywords_augmented(src_lang):
    # fmt: off
    if src_lang == "fr":
        keywords = [
            "french", "france",
            "canada", "canadian",
            "drc",
            "congolese",
        ]  # top3 french speaking country
    elif src_lang == "pl":
        keywords = [
            "polish", "poland",
        ]  # Poland is only country speaking Polish
    elif src_lang == "es":
        keywords = [
            "spanish",          "spain",
            "ecuador",          "ecuadorian",
            "mexico",           "mexican",
            "argentina",        "argentinian",
            "bolivia",          "bolivian",
            "chile",            "chilean",
            "colombia",         "colombian",
            "peru",             "peruvian",
            "uruguay",          "uruguayan",
            "venezuela",        "venezuelan",
            "panama",           "panamanian",
            "cuba",             "cuban",
            "el salvador",      "salvadoran",
            "costa rica",       "costa rican",
            "dominican republic","dominican",
            "puerto rico",      "puerto rican",
            
        ]  # Spain and Mexico is top2 Spanish speaking country, while Ecuador is for XQB
    # fmt: on
    return keywords
