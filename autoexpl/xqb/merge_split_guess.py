import os, sys, time, logging, argparse
from typing import Any, Dict, List, Optional, Tuple, Union
from autoexpl.tools import fileio

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
    parser.add_argument("--ckpt-dir", type=str, help="save printed text", required=False, 
                        default='/13B')
    parser.add_argument("--tasktype", type=str, help="dataset name", required=False, 
                        default='topone_ans_conf') #plqbv1ht512 esqbv1htall
    parser.add_argument("--version-name", type=str, help="dataset name", required=False, 
                        default='pair_coment_charentskip_dedup_gent4')
    parser.add_argument("--lang", type=str, help="source language", required=False, default='es')
    parser.add_argument("--data-dir", type=str, help="alignment data dir", required=False, 
                        default='xqb_eval/extrinsic')
    parser.add_argument("--output-dir", type=str, help="save printed text", required=False, 
                        default='xqb_eval/extrinsic')
    parser.add_argument("--dataset-name", type=str, help="dataset name", required=False, 
                        default='esqbv1htall') #plqbv1ht512 esqbv1htall
    parser.add_argument("--dry-run", help="Dry run", default=False, action='store_true')
    # fmt: on
    return parser.parse_args()


def main(args):
    qanta_filename = os.path.join(
        args.data_dir,
        f"{args.dataset_name}.{args.version_name}.{args.lang}.json",
    )
    qanta_json = fileio.load_singlefile_w_prefix(qanta_filename)
    model_size = os.path.basename(args.ckpt_dir)
    infix = f"LLaMA{model_size}{args.tasktype}.{args.dataset_name}.{args.version_name}.{args.lang}"

    caches = []
    for si, ei in lise:
        final_filename = f"rawresult.{infix}.i{si}-{ei}" + ".pkl"
        final_filepath = os.path.join(
            args.output_dir,
            final_filename,
        )
        result_json = fileio.load_singlefile_w_prefix(final_filepath)
        caches.append(result_json)
    finalcache = {}
    for i, qusjson in enumerate(qanta_json["questions"]):
        exp_id = qusjson["qanta_id"]
        if exp_id == 20054240000:
            aa = 0
        is_result = False
        for cache in caches:
            if exp_id in cache:
                finalcache[exp_id] = cache[exp_id]
                is_result = True
                continue
        assert is_result, f"no guess result {exp_id=}"

    final_filename = f"rawresult.{infix}"
    final_filepath = os.path.join(args.output_dir, final_filename) + ".pkl"
    fileio.save_file(finalcache, final_filepath)


if __name__ == "__main__":
    lise = [
        [0, 1000],
        [1000, 2154],
    ]
    args = parse_args()
    main(args)
