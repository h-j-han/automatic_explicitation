import os, sys, time, logging, argparse, re
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from tqdm import tqdm
import random
from autoexpl.tools import fileio


from autoexpl import utils
from autoexpl.xqb import utils as xqbutils

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s %(message)s",
    # format="%(asctime)s [%(filename)s,%(funcName)s] %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--version-name", type=str, help="dataset name", required=False, 
                        default='pair_coment_charentskip_dedup_gent4')
    parser.add_argument("--src-lang", type=str, help="source language", required=False, default='pl')
    parser.add_argument("--tgt-lang", type=str, help="target language", required=False, default='en')
    parser.add_argument("--data-dir", type=str, help="alignment data dir", required=False, 
                        default='xqb_eval')
    parser.add_argument("--output-dir", type=str, help="save printed text", required=False, 
                        default='xqb_eval')
    parser.add_argument("--dataset-name", type=str, help="dataset name", required=False, 
                        default='plqbv1ht512')
    parser.add_argument("--dry-run", help="Dry run", default=True, action='store_true')
    # fmt: on
    return parser.parse_args()


def find_sent_idx(sents_seg, start_idx):
    # sents_seg=  [[0, 102], [103, 232], [233, 356]]
    for i, (s, e) in enumerate(sents_seg):
        if s <= start_idx and start_idx <= e:
            return i
    assert 0, f" {start_idx=} not included in any sentence. {sents_seg=}"


def display_explicitation(
    explicit_json, explicit_seg_json, l_ent_id_all_gentype_related, src_lang, tgt_lang
):
    lgentype = ["instanceof", "wikides", "wikipara"]
    exps_for_annotators = []
    for i, ent_id in enumerate(l_ent_id_all_gentype_related):
        orig_qid = int(ent_id / 100)
        orig_qus_info = explicit_json["questions"][str(orig_qid)]
        src_sents_seg = explicit_seg_json["questions"][str(orig_qid)]["src_sents_seg"]
        tgt_sents_seg = explicit_seg_json["questions"][str(orig_qid)]["tgt_sents_seg"]

        orig_srcpara = orig_qus_info["srcrawtxt"]
        orig_ent_info = orig_qus_info["per_entities"][str(ent_id)]
        orig_char_start = orig_ent_info["orig_sci"][src_lang]
        orig_char_end = orig_ent_info["orig_eci"][src_lang]
        src_sent_idx = find_sent_idx(src_sents_seg, orig_char_start)
        s, e = src_sents_seg[src_sent_idx]
        disp_src_sent = orig_srcpara[s:e]
        disp_src_undl_idxt = (orig_char_start - s, orig_char_end - s)
        origsline = utils.add_ner_underline_char(
            disp_src_sent, [disp_src_undl_idxt], color_name="GREEN"
        )

        orig_char_start = orig_ent_info["orig_sci"][tgt_lang]
        orig_char_end = orig_ent_info["orig_eci"][tgt_lang]
        tgt_sent_idx = find_sent_idx(tgt_sents_seg, orig_char_start)
        orig_txt = orig_ent_info["orig_txt"][tgt_lang]
        s, e = tgt_sents_seg[tgt_sent_idx]
        assert src_sent_idx == tgt_sent_idx

        #### Manual skip
        if (
            orig_txt.lower() == "poland"
            or orig_txt.lower() == "earth"
            or orig_txt.lower() == "europe"
        ):
            continue
        aa = []
        for j in range(len(lgentype)):
            k = divmod(i + j, len(lgentype))[1]
            gentype = lgentype[k]
            if gentype == "instanceof" and "countryof" in orig_ent_info["gens"]:
                gentype = "countryof"
            exp_id = orig_ent_info["gens"][gentype]["exp_id"]
            if gentype == "wikipara":
                orig_tgtpara = orig_qus_info["tgtrawtxt"]
                disp_tgt_sent = orig_tgtpara[s:e]
                disp_tgt_undl_idxt = (orig_char_start - s, orig_char_end - s)
                source = orig_ent_info["gens"][gentype]["gen_src"][tgt_lang]
                # source = re.sub(r'\(.*\)', "", source)
                sourceline = f"*{orig_txt} : {source}.."

            else:
                exp_tgtpara = orig_ent_info["gens"][gentype]["exp_rawtxt"][tgt_lang]
                exp_char_start = orig_ent_info["gens"][gentype]["exp_sci"][tgt_lang]
                exp_char_end = orig_ent_info["gens"][gentype]["exp_eci"][tgt_lang]
                exp_char_diff = exp_char_end - orig_char_end
                assert exp_char_diff
                e += exp_char_diff
                disp_tgt_sent = exp_tgtpara[s:e]
                disp_tgt_undl_idxt = (exp_char_start - s, exp_char_end - s)
                tline = utils.add_ner_underline_char(
                    disp_tgt_sent, [disp_tgt_undl_idxt], color_name="BLUE"
                )
                sourceline = ""
            tline = utils.add_ner_underline_char(
                disp_tgt_sent, [disp_tgt_undl_idxt], color_name="BLUE"
            )
            tline += "\n"
            tline += sourceline
            one_ent_one_gen = {
                "orig_qid": orig_qid,
                "ent_id": ent_id,
                "gentype": gentype,
                "exp_id": exp_id,
                "idx_ent_id_all_gentype_related": i,  # l_ent_id_all_gentype_related
                "src_sent": disp_src_sent,
                "src_undl_idxt": disp_src_undl_idxt,
                "tgt_sent": disp_tgt_sent,
                "tgt_undl_idxt": disp_tgt_undl_idxt,
                "footnote": sourceline,
            }
            aa.append(one_ent_one_gen)
        exps_for_annotators.append(aa)
    random.shuffle(exps_for_annotators)
    # Check for first annotator
    for exp_for_annotators in exps_for_annotators:
        exp_json = exp_for_annotators[0]
        orig_qid = exp_json["orig_qid"]
        ent_id = exp_json["ent_id"]
        gentype = exp_json["gentype"]
        exp_id = exp_json["exp_id"]

        disp_src_sent = exp_json["src_sent"]
        disp_src_undl_idxt = exp_json["src_undl_idxt"]
        disp_tgt_sent = exp_json["tgt_sent"]
        disp_tgt_undl_idxt = exp_json["tgt_undl_idxt"]
        sourceline = exp_json["footnote"]

        origsline = utils.add_ner_underline_char(
            disp_src_sent, [disp_src_undl_idxt], color_name="GREEN"
        )
        tline = utils.add_ner_underline_char(
            disp_tgt_sent, [disp_tgt_undl_idxt], color_name="BLUE"
        )
        print(f"{gentype=} {ent_id=} {orig_qid=}")
        print(f"{origsline}\n{tline}\n{sourceline}\n")
    return exps_for_annotators


def main(args):
    exp_filename = os.path.join(
        f"{args.dataset_name}.{args.version_name}.{args.src_lang}{args.tgt_lang}.exp.json",
    )
    # seg_file: this is same one with exp_file
    # However,
    # Especially for wikipara generation, the source changed a lot between several months (wiki has active editions).
    # so only use the source segmentation
    # future version than pair_coment_charentskip_dedup_gent4 will have all the integrated ones.
    exp_seg_filename = os.path.join(
        "extrinsic",
        f"{args.dataset_name}.{args.version_name}si.{args.src_lang}{args.tgt_lang}.exp.json",
    )
    explicit_json = fileio.load_singlefile_w_prefix(
        os.path.join(args.data_dir, exp_filename)
    )
    explicit_seg_json = fileio.load_singlefile_w_prefix(
        os.path.join(args.data_dir, exp_seg_filename)
    )

    # fmt: off
    l_ent_id_all_gentype_related =[
        200001904, 200002900, 200002901, 200002902, 200003501, 200007103, 200008003, 200500801, 200500802, 200500803, 200504404, 200505703, 200506900, 200506901, 200509601, 200509602, 200509604, 200510201, 200510504, 200510703, 200511302, 200520900, 200521700, 200521702, 200522303, 200522304, 200522305, 200524803, 200525203, 200525606, 200526001, 200529802, 200529803, 200529804, 200529805, 200529806, 200531301, 200537401, 200537402, 200538003, 200543402, 200543403, 200543404, 200544203, 200548401, 200548702, 200554700, 200561000, 200593102
    ] 
    # fmt: on
    exps_for_annotators = display_explicitation(
        explicit_json,
        explicit_seg_json,
        l_ent_id_all_gentype_related,
        args.src_lang,
        args.tgt_lang,
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
