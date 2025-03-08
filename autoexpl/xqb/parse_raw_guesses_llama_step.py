import os, sys, time, logging, argparse
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from autoexpl.tools import fileio
from autoexpl.xqb import utils
from autoexpl.xqb import gen_prompt
from autoexpl.xqb.utils import qidmap_lqus2dict

import ast
import re


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
    parser.add_argument("--ckpt-dir", type=str, help="model_size", required=False, 
                        default='/13B')
    parser.add_argument("--word-skip", type=int, default=46, required=False, 
                        help='wordskip value for words_entity_skip funcion')
    parser.add_argument("--nguess", type=int, default=1, required=False, 
                        help='num of nguess')
    parser.add_argument("--version-name", type=str, help="dataset name", required=False, 
                        default='pair_coment_charentskip_dedup_gent4')
    parser.add_argument("--tasktype", type=str, help="dataset name", required=False, 
                        default='topone_ans_conf') #plqbv1ht512 esqbv1htall
    parser.add_argument("--lang", type=str, help="source language", required=False, default='pl')
    parser.add_argument("--data-dir", type=str, help="alignment data dir", required=False, 
                        default='xqb_eval/extrinsic')
    parser.add_argument("--output-dir", type=str, help="save printed text", required=False, 
                        default='xqb_eval/extrinsic')
    parser.add_argument("--reporting-dir", type=str, help="save printed text", required=False, 
                        default='xqb_eval/extrinsic')
    parser.add_argument("--dataset-name", type=str, help="dataset name", required=False, 
                        default='plqbv1ht512') #plqbv1ht512 esqbv1htall
    parser.add_argument("--dry-run", help="Dry run", default=False, action='store_true')
    # fmt: on
    return parser.parse_args()


def parse_llama_output(input_string, nguess, filler):
    # Regular expression pattern to match pairs of a string and a float
    pattern = r'["\']([^"\']+)["\'],\s*([\d.]+)'

    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, input_string)
    matches = matches[:nguess]
    # Convert the second element of each match to a float
    try:
        pairs = [
            (country.replace("(", "").replace(")", ""), float(value))
            for country, value in matches
        ]
    except Exception as e:
        logger.info(f"ERRMSG:{e}\n{matches=}")
        if len(matches) > 0:  # recycle the cases matches=[('Moby-Dick', '...')]
            pairs = [(matches[0][0], filler[1])]
            logger.info(f"recycle single guesses {pairs=}")
        else:
            pairs = []
    return pairs


def parse_raw_guess(
    cache,
    qanta_json,
    word_skip,
    lang,
    nguess,
    tasktype="topone_ans_conf",
    filler=("unknown", 0.001),
):

    qidmap = qidmap_lqus2dict(qanta_json["questions"])
    gathers = {}
    total_counter = 0
    incomplete_counter = 0
    for i, qusjson in enumerate(qanta_json["questions"]):
        exp_id = qusjson["qanta_id"]
        assert exp_id in cache
        orig_id = qusjson["qdb_id"]
        # if orig_id in lorig_id_entorder_before:
        #     continue

        # Baching
        text = qusjson["text"]
        tokenizations = qusjson["tokenizations"]
        char_skip = qusjson["trick_id"]
        lsubtexts, lindices = utils.char_entity_skip(text, char_skip, tokenizations)

        origqus = qidmap[orig_id]
        clsubtexts, clindices = utils.char_entity_skip(
            origqus["text"], origqus["trick_id"], origqus["tokenizations"]
        )
        assert len(lindices) == len(clindices)
        if "hint" in tasktype:
            lindices = clindices

        new_indices = []
        for tokenization in tokenizations:
            s, e = tokenization
            assert s in lindices, f"{orig_id=} {exp_id=} {tokenization=} {lindices=}"
            assert (
                e + 1 in lindices
            ), f"{orig_id=} {exp_id=} {tokenization=} {lindices=}"

            new_indices.append(e + 1)
        new_indices.append(lindices[-1])
        gathers[exp_id] = []
        for step in new_indices:
            # for step in cache[exp_id]:
            # for j, step in enumerate(lindices):
            total_counter += 1
            assert step in lindices
            assert step in cache[exp_id]
            gather = {}
            # prompt = prompts[step]
            rawoutput = cache[exp_id][step]
            # if prompt in rawoutput:
            #     rawoutput = rawoutput.replace(prompt, "").strip()
            pairs = parse_llama_output(rawoutput, nguess, filler)
            # Complement
            if len(pairs) < nguess:
                incomplete_counter += 1
                ncomp = nguess - len(pairs)
                # logger.info(
                #     f"{exp_id=} {step=} Not enough guesses {len(pairs)=} Complement {ncomp=} with unknowns"
                # )
                for i in range(ncomp):
                    pairs.append(filler)
            assert len(pairs) == nguess
            # print(f"{exp_id=} {step=} {wanted=}")
            guesses = [guess for guess, score in pairs]
            scores = [score for guess, score in pairs]
            gather["char_index"] = step
            gather["guess"] = guesses[0]
            gather["guesses"] = guesses
            gather["score"] = scores[0]
            gather["scores"] = scores

            gathers[exp_id].append(gather)
    logger.info(
        f"{total_counter=} {incomplete_counter=} {incomplete_counter/total_counter*100:0.2f}%"
    )
    return gathers


def main(args):
    qanta_filename = os.path.join(
        args.data_dir,
        f"{args.dataset_name}.{args.version_name}.{args.lang}.json",
    )
    qanta_json = fileio.load_singlefile_w_prefix(qanta_filename)
    model_size = os.path.basename(args.ckpt_dir)
    final_filename = f"rawresult.LLaMA{model_size}{args.tasktype}.{args.dataset_name}.{args.version_name}.{args.lang}"
    final_filepath = os.path.join(args.output_dir, final_filename) + ".pkl"
    cache = fileio.load_singlefile_w_prefix(final_filepath)
    gathers = parse_raw_guess(
        cache,
        qanta_json,
        args.word_skip,
        args.lang,
        args.nguess,
        tasktype=args.tasktype,
    )
    if not args.dry_run:
        gather_file = os.path.join(
            args.reporting_dir,
            f"LLaMA{model_size}{args.tasktype}Guesser.{args.dataset_name}.{args.version_name}_stepreduced.{args.lang}.gather.pickle",
        )
        fileio.save_file(gathers, gather_file)


if __name__ == "__main__":
    args = parse_args()
    main(args)
