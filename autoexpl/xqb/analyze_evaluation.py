import os, sys, argparse, json, time, logging, re
from typing import List, Iterable, Mapping, Tuple, Dict, Union, Optional, Any
import numpy as np
from autoexpl.tools import fileio
from autoexpl import utils

NUM_EVAL = 42

logger = logging.getLogger(__name__)
logging.basicConfig(
    stream=sys.stdout,
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s %(message)s",
    # format="%(asctime)s [%(filename)s,%(funcName)s] %(message)s",
)


def get_detect_exp_name_from_args(args):
    infix = utils.get_detect_exp_name(
        args.dataset_name,
        args.src_lang + args.tgt_lang,
        args.pretokname,
        args.ner_name,
        args.aligner_names,
        args.vote_threshold,
        args.proximity_threshold,
    )
    return infix


def parse_args():
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("--version-name", type=str, help="dataset name", required=False, 
                        default='pair_coment_charentskip_dedup_gent4')
    parser.add_argument("--sheet-version", type=str, help="dataset name", required=False, 
                        default='v1')
    parser.add_argument("--src-lang", type=str, help="source language", required=False, default='pl')
    parser.add_argument("--tgt-lang", type=str, help="target language", required=False, default='en')
    parser.add_argument("--data-dir", type=str, help="save printed text", required=False, 
                        default='xqb_eval/intrinsic')
    parser.add_argument("--output-dir", type=str, help="save printed text", required=False, 
                        default='xqb_eval/intrinsic')
    parser.add_argument("--n-annotators", type=str, required=False, default=3,
                        help="number of annotators anootated one example",)
    parser.add_argument("--dry-run", help="Dry run", default=False, action='store_true')
    # fmt: on
    return parser.parse_args()


d2i = {
    "Yes helpful, this is well-known to Polish speaker and the target audience might not know this.": 5,
    "Might be helpful, but this is not also well-known to the Polish speaker, too.": 4,
    "No need. Already sufficiently explained in the original text.": 3,
    "No need. The target audience may already know this.": 2,
    "etc": 1,
}
e2i = {
    "Appropriate and well written explanation for explicitation.": 4,
    "Related, but not very helpful. (e.g. too obvious or even make complicated)": 3,
    "Inappropriate/wrong explanation (e.g. mismatch btw entity and explanation)": 2,
    "etc": 1,
}
i2i = {
    "Smoothly integrated, and similar readability": 4,
    "Understandable but integration is not smooth": 3,
    "Introducing incorrect grammar or confusion": 2,
    "etc": 1,
}


def analyze_evaluationss(evaluations):
    ent_ids = evaluations[2]["ent2exp_ids"].keys()
    laidx = [1, 2, 0]
    stats = {
        "decision": {
            0: {
                5: 0,  # yes helpful
                4: 0,  # maybe helpful
                3: 0,  # no need already in sent
                2: 0,  # no need too general
                1: 0,  # etc
            },
            1: {
                5: 0,  # yes helpful
                4: 0,  # maybe helpful
                3: 0,  # no need already in sent
                2: 0,  # no need too general
                1: 0,  # etc
            },
            2: {
                5: 0,  # yes helpful
                4: 0,  # maybe helpful
                3: 0,  # no need already in sent
                2: 0,  # no need too general
                1: 0,  # etc
            },
        },
        "explanation": {
            "long": {
                4: 0,  # App
                3: 0,  # Related
                2: 0,  # Inapp
                1: 0,  # etc
            },
            "mid": {
                4: 0,  # App
                3: 0,  # Related
                2: 0,  # Inapp
                1: 0,  # etc
            },
            "short": {
                4: 0,  # App
                3: 0,  # Related
                2: 0,  # Inapp
                1: 0,  # etc
            },
        },
        "integration": {
            "long": {
                4: 0,  # App
                3: 0,  # Related
                2: 0,  # Inapp
                1: 0,  # etc
            },
            "mid": {
                4: 0,  # App
                3: 0,  # Related
                2: 0,  # Inapp
                1: 0,  # etc
            },
            "short": {
                4: 0,  # App
                3: 0,  # Related
                2: 0,  # Inapp
                1: 0,  # etc
            },
        },
    }

    annotates = {
        0: [],
        1: [],
        2: [],
    }
    for ent_id in ent_ids:
        # decision
        vals = []
        for aidx in laidx:
            assert ent_id in evaluations[aidx]["ent2exp_ids"]
            exp_id = evaluations[aidx]["ent2exp_ids"][ent_id]
            val = d2i[evaluations[aidx]["exp_ids"][str(exp_id)]["decision"]]
            stats["decision"][aidx][val] += 1
            vals.append(val)
            if val in [5, 4, 1]:
                annotates[aidx].append(True)
            else:
                annotates[aidx].append(False)

        # generation & integration
        for aidx in laidx:
            exp_id = evaluations[aidx]["ent2exp_ids"][ent_id]
            if exp_id % 10 == 2:
                gtype = "long"
            elif exp_id % 10 == 0:
                gtype = "mid"
            else:
                gtype = "short"
            val = e2i[evaluations[aidx]["exp_ids"][str(exp_id)]["explanation"]]
            stats["explanation"][gtype][val] += 1

            val = i2i[evaluations[aidx]["exp_ids"][str(exp_id)]["integration"]]
            stats["integration"][gtype][val] += 1
    # Decision scoring
    # annotators mostly annotated "etc" for the cases where the decision to do explicitation is right
    # but generated explicitation or integration is wrong.
    # This is not intended behavior but adding etc as correct decision
    # we score decision as binary
    avg = 0
    for key in stats["decision"]:
        ntot = sum([stats["decision"][key][v] for v in stats["decision"][key]])
        npos = (
            stats["decision"][key][5]
            + stats["decision"][key][4]
            + stats["decision"][key][1]
        )
        avg += npos / ntot
        print(
            f'Annotator {key}, {npos=} {ntot=} {(stats["decision"][key][5] + stats["decision"][key][4] + stats["decision"][key][1])/ntot}'
        )
    print(f"Decision {avg/3}")
    # Explanation and Integration scoring
    # there are three version of explicitation to be scored for each entity.
    # we divided each one randomly to each annotators
    # we gather all from three annotators and score
    for key in ["explanation", "integration"]:
        for gtype in stats[key]:
            ntot = sum([stats[key][gtype][v] for v in stats[key][gtype]])
            print(
                f"{key} {gtype=} {ntot=} {(stats[key][gtype][4] + stats[key][gtype][3] * 0.5 + stats[key][gtype][1])/ntot}"
            )


if __name__ == "__main__":
    args = parse_args()
    results = {}
    for annotator_idx in [0, 1, 2]:
        gsheet_name = f"{args.version_name}.{args.src_lang}{args.tgt_lang}.shuf.anidx{annotator_idx}.{args.sheet_version}"
        result_filename = os.path.join(
            f"eval_results.{gsheet_name}.json",
        )
        result = fileio.load_singlefile_w_prefix(
            os.path.join(args.output_dir, result_filename)
        )
        results[annotator_idx] = result
    analyze_evaluationss(results)
