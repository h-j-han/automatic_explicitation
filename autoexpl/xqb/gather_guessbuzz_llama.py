import pandas as pd
import numpy as np

from tqdm import tqdm
import os, sys, argparse, json, time, logging, re
from typing import List, Iterable, Mapping, Tuple, Dict, Union, Optional, Any

from autoexpl.tools import fileio

from autoexpl.xqb.utils import (
    get_mrr_steps,
    get_mrr_step,
    get_log_mrr_steps,
    answer_matched,
    CurveScore,
)
from autoexpl.utils import lidx_tok2char
from autoexpl.xqb.utils import qidmap_lqus2dict


logger = logging.getLogger("")
logging.basicConfig(
    stream=sys.stdout,
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s %(message)s",
)


def parse_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--version-name", type=str, help="dataset name", required=False, 
                        default='pair_coment_charentskip_dedup_gent4')
    parser.add_argument("--guessers", type=str, help="dataset name", required=False, nargs="*",
                        default=["LLaMA13Btopone_ans_confGuesser","LLaMA7Btopone_ans_confGuesser"])
    parser.add_argument("--buzz-threshold", type=float, required=False,  nargs="*",
                        default=[0.7, 0.5])
    parser.add_argument("--dataset-names", type=str, help="dataset name", required=False, nargs="*",
                        default=['plqbv1ht512', 'esqbv1htall'])
    parser.add_argument("--reporting-dir", type=str, help="save printed text", required=False, 
                        default='xqb_eval/extrinsic')
    parser.add_argument("--data-dir", type=str, help="save printed text", required=False, 
                        default='xqb_eval/extrinsic')
    parser.add_argument("--output-dir", type=str, help="save printed text", required=False, 
                        default='xqb_eval/extrinsic')
    parser.add_argument("--dry-run", help="Dry run", default=False, action='store_true')
    # fmt: on
    return parser.parse_args()


def gather_guessbuzz(tot_dfs, qanta_json, explicit_json, lang, guesser, buzz_threshold):
    cs = CurveScore()
    # remove duplicate qus
    lorig_id_entorder_before = [
        2000079,
        2005234,
        2005318,
        2005047,
        2005243,
        2005459,
        3000064,
    ]
    # fmt: off
    df = pd.DataFrame(columns = ['exp_qid','ent_id','orig_qid', 'gentype', 
                                 'orig_mrr_eci', 'exp_mrr_eci', 'orig_mrr_end', 'exp_mrr_end'
                                 'orig_mrr_avg', 'exp_mrr_avg', 'orig_lmrr_avg', 'exp_lmrr_avg', 'orig_mrr_max', 'exp_mrr_max'
                                 'orig_ew', 'orig_ewo', 'exp_tew', 'exp_tewo', 'exp_ew', 'exp_ewo', 
                                 ])
    # fmt: on
    qidmap = qidmap_lqus2dict(qanta_json["questions"])
    l_exp_qid = list(explicit_json["expid2prop"].keys())  # str key
    data = []
    for exp_qid in tqdm(l_exp_qid):  # key str
        exp_qid = int(exp_qid)
        orig_qid = explicit_json["expid2prop"][str(exp_qid)]["orig_id"]
        if orig_qid in lorig_id_entorder_before:
            continue
        assert exp_qid != orig_qid
        ent_id = explicit_json["expid2prop"][str(exp_qid)]["ent_id"]
        gentype = explicit_json["expid2prop"][str(exp_qid)]["gentype"]
        # per orig qus
        orig_qus_info = explicit_json["questions"][str(orig_qid)]
        dataset = orig_qus_info["dataset"]
        answer = orig_qus_info["answer"]
        answerpair = id2rawans[orig_qid]
        orig_srcpara = orig_qus_info["srcrawtxt"]
        orig_tgtpara = orig_qus_info["tgtrawtxt"]
        # per ent
        orig_ent_info = orig_qus_info["per_entities"][str(ent_id)]
        orig_sci = orig_ent_info["orig_sci"][lang]
        orig_eci = orig_ent_info["orig_eci"][lang]
        exp_sci = orig_ent_info["gens"][gentype]["exp_sci"][lang]
        exp_eci = orig_ent_info["gens"][gentype]["exp_eci"][lang]
        exp_tgtpara = orig_ent_info["gens"][gentype]["exp_rawtxt"][lang]
        # guesses orig vs exp
        orig_full_input_top_guess = tot_dfs[orig_qid][-1]["guess"]
        exp_full_input_top_guess = tot_dfs[exp_qid][-1]["guess"]
        orig_char_idxs = [guess["char_index"] for guess in tot_dfs[orig_qid]]
        exp_char_idxs = [guess["char_index"] for guess in tot_dfs[exp_qid]]
        assert len(tot_dfs[orig_qid]) == len(tot_dfs[exp_qid])
        assert orig_eci + 1 in orig_char_idxs or orig_eci in orig_char_idxs
        if "hint" not in guesser:
            assert exp_eci + 1 in exp_char_idxs or exp_eci in exp_char_idxs
        orig_gus_pos = orig_char_idxs.index(orig_eci + 1)  # for word split
        # orig_gus_pos = orig_char_idxs.index(orig_eci)  # for char split
        if "hint" not in guesser:
            exp_gus_pos = exp_char_idxs.index(exp_eci + 1)  # for word split
        # exp_gus_pos = exp_char_idxs.index(exp_eci)  # for char split
        # if fillped, then position will be different but not much
        if "hint" not in guesser:
            assert orig_gus_pos == exp_gus_pos
        gus_pos = orig_gus_pos

        orig_mrrs = get_mrr_steps(tot_dfs[orig_qid], answerpair)
        exp_mrrs = get_mrr_steps(tot_dfs[exp_qid], answerpair)
        orig_idx_info_dict = {
            "src": orig_tgtpara if lang == "en" else orig_srcpara,
            "tgtdelay": [i for i in range(len(tot_dfs[exp_qid]))],
            "srccharidx": [guess["char_index"] for guess in tot_dfs[orig_qid]],
        }
        data.append(
            {
                "dataset": dataset,
                "guesser": guesser,
                "lang": lang,
                "exp_qid": exp_qid,
                "ent_id": ent_id,
                "orig_qid": orig_qid,
                "gentype": gentype,
                "orig_mrr_eci": orig_mrrs[gus_pos],
                "exp_mrr_eci": exp_mrrs[gus_pos],
                "orig_full_acc": answer_matched(orig_full_input_top_guess, answerpair),
                "exp_full_acc": answer_matched(exp_full_input_top_guess, answerpair),
                "orig_mrr_end": orig_mrrs[-1],
                "exp_mrr_end": exp_mrrs[-1],
                "orig_mrr_avg": np.mean(orig_mrrs),
                "exp_mrr_avg": np.mean(exp_mrrs),
                "orig_lmrr_avg": np.mean(get_log_mrr_steps(orig_mrrs)),
                "exp_lmrr_avg": np.mean(get_log_mrr_steps(exp_mrrs)),
                "orig_mrr_max": np.max(orig_mrrs),
                "exp_mrr_max": np.max(exp_mrrs),
                "orig_ew": cs.score_orig(
                    tot_dfs[orig_qid],
                    qidmap[orig_qid],
                    threshold=buzz_threshold,
                    newans=answerpair,
                ),
                "orig_ewo": cs.score_optimal(
                    tot_dfs[orig_qid], qidmap[orig_qid], newans=answerpair
                ),
                "exp_ew": cs.score_orig(
                    tot_dfs[exp_qid],
                    qidmap[exp_qid],
                    threshold=buzz_threshold,
                    newans=answerpair,
                ),
                "exp_ewo": cs.score_optimal(
                    tot_dfs[exp_qid], qidmap[exp_qid], newans=answerpair
                ),
                "exp_tew": cs.trans_score2(
                    tot_dfs[exp_qid],
                    qidmap[exp_qid],
                    orig_idx_info_dict,
                    threshold=buzz_threshold,
                    newans=answerpair,
                ),
                "exp_tewo": cs.trans_score_optimal2_exp(
                    tot_dfs[exp_qid],
                    qidmap[exp_qid],
                    orig_idx_info_dict,
                    newans=answerpair,
                ),
            }
        )
    logger.info(f"{len(data)=}")
    df = pd.DataFrame(data)

    return data


def get_langs_from_dataset(dataset):
    if dataset == "plqbv1ht512":
        langs = ["pl", "en"]
    elif dataset == "esqbv1htall":
        langs = ["es", "en"]
    return langs


def main(args):
    version_name = args.version_name
    data = []
    for buzz_threshold, dataset, guesser in zip(
        args.buzz_threshold, args.dataset_names, args.guessers
    ):
        langs = get_langs_from_dataset(dataset)
        pair = "".join(langs)
        for lang in langs:
            gather_file = os.path.join(
                args.reporting_dir,
                f"{guesser}.{dataset}.{version_name}_stepreduced.{lang}.gather.pickle",
            )
            if not os.path.isfile(gather_file):
                logger.info(
                    f"{dataset=} {lang=} {guesser=} File does not exist. pass. {gather_file}"
                )
                continue
            qanta_json_file = os.path.join(
                args.data_dir, f"{dataset}.{version_name}.{lang}.json"
            )
            exp_json_file = os.path.join(
                args.data_dir, "../", f"{dataset}.{version_name}.{pair}.exp.json"
            )
            tot_dfs = pd.read_pickle(gather_file)
            qanta_json = fileio.load_file(qanta_json_file)
            explicit_json = fileio.load_file(exp_json_file)
            data += gather_guessbuzz(
                tot_dfs,
                qanta_json,
                explicit_json,
                lang,
                guesser,
                buzz_threshold,
            )
    if not args.dry_run:
        output_file = os.path.join(
            args.output_dir,
            f'origvsexp.LLaMAtopone_ans_confGuesser.{"-".join(args.dataset_names)}.{version_name}_stepreduced.srcans.pickle',
        )
        logger.info(f"Write pickle file to {output_file}")
        df = pd.DataFrame(data)
        df.to_pickle(output_file)


if __name__ == "__main__":
    # e.g. dict 2000018: {'pl': 'GRECJA', 'en': 'Greece'}
    id2rawans = fileio.load_file("xqb_eval/extrinsic/id2rawans.pickle")
    args = parse_args()
    main(args)
