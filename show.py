import json
from typing import List
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

def tup_range_to_idx(tupleidxrange):
    slne = []
    for t in tupleidxrange:
        s = t[0]
        e = t[1]
        for i in range(s, e + 1):
            slne.append(i)
    return slne

def add_highlight_tokidx(src_tok: List[str], slne: List[int], color_name="GREEN") -> str:
    sline = ""
    for i, x in enumerate(src_tok):
        if i in slne:
            sline += f"{color.BOLD}{getattr(color, color_name)}{color.RED}{x}{color.END} "
        else:
            sline += f"{getattr(color, color_name)}{x}{color.END} "
    return sline
def print_span(lsrcbrack, srctok_orig, color):
    slne = tup_range_to_idx(lsrcbrack)
    sline =add_highlight_tokidx(srctok_orig, slne, color_name=color)
    return sline

def print_all_wikiexpl(data):
    expl_idx_list = data['expl_idx_list']
    for i, idx in enumerate(expl_idx_list):
        idx = str(idx)
        print_wikiexpl(data, idx)

def print_wikiexpl(data, idx):
        srctok_orig = data["candidates"][idx]["src_toks"]
        tgttok_orig = data["candidates"][idx]["tgt_toks"]
        lsrcbrack = data["candidates"][idx]["src_toks_span"]
        ltgtbrack = data["candidates"][idx]["tgt_toks_span"]
        print(f"idx: {idx}")
        print(f'FR: {print_span(lsrcbrack, srctok_orig,  "GREEN")}')
        print(f'EN with explicitation : {print_span(ltgtbrack, tgttok_orig,  "BLUE")}')
        print("")

        
def main():
    data_path ="wikiexpl/explicitations.france.json"
    with open(data_path, "r") as f:
        wikiexpl = json.load(f)
    # print_all_wikiexpl(wikiexpl)
    print_wikiexpl(wikiexpl, "11516")
    print_wikiexpl(wikiexpl, "23139")
    
if __name__ == "__main__":
    main()
