import os, sys, time, logging, argparse
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Union
from autoexpl.tools import fileio
from autoexpl.tools import llama_lib as llama
from autoexpl.xqb import utils
from autoexpl.xqb import gen_prompt

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
                        default=os.path.join(os.getenv("HOME"),'gits/llama/llama_weights/7B'))
    parser.add_argument("--max-seq-len", type=int, default=650, required=False, 
                        help='llama max seq len of the model input.')
    parser.add_argument("--max-batch-size", type=int, default=32, required=False, 
                        help='llama max batch size of the model input.')
    parser.add_argument("--word-skip", type=int, default=46, required=False, 
                        help='wordskip value for words_entity_skip funcion')
    parser.add_argument("--nguess", type=int, default=1, required=False, 
                        help='num of nguess')
    parser.add_argument("--start", type=int, default=0, required=False, 
                        help='num of nguess') #1895
    parser.add_argument("--end", type=int, default=5, required=False, 
                        help='num of nguess') #3789
    parser.add_argument("--version-name", type=str, help="dataset name", required=False, 
                        default='pair_coment_charentskip_dedup_gent4')
    parser.add_argument("--lang", type=str, help="source language", required=False, default='en')
    parser.add_argument("--data-dir", type=str, help="alignment data dir", required=False, 
                        default='xqb_eval/extrinsic')
    parser.add_argument("--output-dir", type=str, help="save printed text", required=False, 
                        default='xqb_eval/extrinsic')
    parser.add_argument("--dataset-name", type=str, help="dataset name", required=False, 
                        default='plqbv1ht512') #plqbv1ht512 esqbv1htall
    parser.add_argument("--dry-run", help="Dry run", default=False, action='store_true')
    # fmt: on
    return parser.parse_args()


# def merge_expid_charidx(exp_id, char_index):
#     return f"exp_id{exp_id}_charidx{char_index}"


def generate_guess(
    cache,
    qanta_json,
    word_skip,
    lang,
    nguess,
    ckpt_dir,
    max_seq_len,
    max_batch_size,
    final_filepath,
    n_update_to_save=25,
    start=0,
    end=-1,
    tasktype="topone_ans_conf",
    temperature: float = 0.8,
    top_p: float = 0.95,
    ratio_word2sub: float = 1.4,
):
    generator = llama.get_llama_generator(
        ckpt_dir=ckpt_dir, max_seq_len=max_seq_len, max_batch_size=max_batch_size
    )
    ntotal = len(qanta_json["questions"])
    update_cnt = 0
    if end < 0:
        end = ntotal
    for i in range(start, end):
        qusjson = qanta_json["questions"][i]
        exp_id = qusjson["qanta_id"]
        if exp_id not in cache:
            orig_id = qusjson["qdb_id"]

            # Baching
            text = qusjson["text"]
            tokenizations = qusjson["tokenizations"]
            char_skip = qusjson["trick_id"]
            lsubtexts, lindices = utils.char_entity_skip(text, char_skip, tokenizations)
            prompts = []
            max_gen_len = max_seq_len
            for subtexts in lsubtexts:
                prompt = gen_prompt.gen_one_prompt(subtexts, lang, tasktype=tasktype)
                estnsubw = len(prompt.split()) * ratio_word2sub + 10 * nguess
                if max_gen_len < estnsubw:
                    max_gen_len = estnsubw
                prompts.append(prompt)

            # Generate
            tmptime = time.time()
            results = generator.generate(
                prompts, max_gen_len=max_gen_len, temperature=temperature, top_p=top_p
            )
            logger.info(
                f"Success {i}/{ntotal} {exp_id=} takes {time.time() - tmptime:.3f} sec. {len(results)=}"
            )

            # Save
            assert len(prompts) == len(results)
            gather = {}
            for k, result in enumerate(results):
                # gen_prompt.parse_llama_output(result, prompts[i])
                # Currently just save the raw output
                prompt = prompts[k]
                gather[lindices[k]] = result.replace(prompt, "").strip()
            cache[exp_id] = gather
            if n_update_to_save <= update_cnt:
                fileio.save_file(cache, final_filepath)
                update_cnt = 0
            else:
                update_cnt += 1
        else:
            logger.info(f"Skip {i}/{ntotal} {exp_id=} as already have results")


def main(args):
    qanta_filename = os.path.join(
        args.data_dir,
        f"{args.dataset_name}.{args.version_name}.{args.lang}.json",
    )
    qanta_json = fileio.load_singlefile_w_prefix(qanta_filename)
    model_size = os.path.basename(args.ckpt_dir)
    final_filename = f"rawresult.LLaMA{model_size}.{args.dataset_name}.{args.version_name}.{args.lang}.i{args.start}-{args.end}"
    final_filepath = os.path.join(args.output_dir, final_filename) + ".pkl"
    if os.path.isfile(final_filepath):
        cache = fileio.load_singlefile_w_prefix(final_filepath)
    else:
        logger.info(f"Starting from scratch. No mid result found")
        cache = {}
    initaln = len(cache)
    try:
        generate_guess(
            cache,
            qanta_json,
            args.word_skip,
            args.lang,
            args.nguess,
            args.ckpt_dir,
            args.max_seq_len,
            args.max_batch_size,
            final_filepath,
            start=args.start,
            end=args.end,
        )
    except Exception as e:
        currentn = len(cache)
        logger.info(
            f"INTERRUPTED. Added {currentn=} - {initaln=} = {currentn - initaln}"
        )
        logger.info(f"\tERRMSG: {e}")
    fileio.save_file(cache, final_filepath)


if __name__ == "__main__":
    args = parse_args()
    main(args)
