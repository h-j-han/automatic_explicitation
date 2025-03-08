import argparse
import logging, time
import pickle
import json, os, sys
import numpy as np

logger = logging.getLogger(__name__)


def load_singlefile_w_prefix(pathfilenameprefix, suffix=""):
    import glob

    candlist = glob.glob(f"{pathfilenameprefix}*{suffix}")
    if len(candlist) == 0:
        logger.info(f"{pathfilenameprefix}")
        raise FileNotFoundError
    if len(candlist) > 1:
        logger.info(f"{candlist}")
        raise NotImplementedError
    return load_file(candlist[0])


def load_file(pathfilename):
    tmptime = time.time()
    if ".pickle" == pathfilename[-7:] or ".pkl" == pathfilename[-4:]:
        import pickle

        with open(pathfilename, "rb") as f:
            output = pickle.load(f)
    elif ".json" == pathfilename[-5:]:
        import json

        with open(pathfilename, "r") as f:
            output = json.load(f)
    else:
        output = []
        with open(pathfilename, "r") as f:
            time0 = time.time()
            for i, line in enumerate(f):
                output.append(line.strip())
                if i > 0 and i % 10000 == 0:
                    logger.info(
                        "Read text file line %d with %f sec" % (i, time.time() - time0)
                    )
                    time0 = time.time()
    logger.info(
        f"Loaded file succesfully in {time.time() - tmptime:.3f}sec, size: {sizeof_fmt(os.path.getsize(pathfilename))}, loc: {pathfilename} "
    )
    return output


def load_paired_files(lang1_file, lang2_file):
    with open(lang1_file, "r") as f1, open(lang2_file, "r") as f2:
        line_num1 = sum(1 for _ in f1)
        line_num2 = sum(1 for _ in f2)
        assert line_num1 == line_num2, "%d lines in %s \n%d lines in %s" % (
            line_num1,
            lang1_file,
            line_num2,
            lang2_file,
        )
        f1.seek(0)
        f2.seek(0)
        time0 = time.time()
        line1s = []
        line2s = []
        for i, (line1, line2) in enumerate(zip(f1, f2)):
            line1s.append(line1.strip())
            line2s.append(line2.strip())
            if i > 0 and i % 10000 == 0:
                logger.info(
                    "Read text file line %d with %f sec" % (i, time.time() - time0)
                )
                time0 = time.time()
    return line1s, line2s


def sizeof_fmt(num, suffix="B"):
    # "too small a task to require a library" issue
    # https://stackoverflow.com/questions/1094841/get-human-readable-version-of-file-size
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def save_file(things, pathfilename, not_pretty=False):
    tmptime = time.time()
    if ".pickle" == pathfilename[-7:] or ".pkl" == pathfilename[-4:]:
        import pickle

        with open(pathfilename, "wb") as f:
            output = pickle.dump(things, f)
    elif ".json" == pathfilename[-5:]:
        import json

        if not_pretty:
            with open(pathfilename, "w") as f:
                output = json.dump(things, f)
        else:
            with open(pathfilename, "w", encoding="utf-8") as f:
                output = json.dump(things, f, ensure_ascii=False, indent=4)
                # json.dump(outputjson, fo, ensure_ascii=False, indent=4)
    else:
        raise NotImplementedError
    logger.info(
        f"Saved file succesfully in {time.time() - tmptime:.3f}sec, size: {sizeof_fmt(os.path.getsize(pathfilename))}, loc: {pathfilename} "
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data-dir",
        type=str,
        help="Data directory",
        required=False,
        default="/cliphomes/hjhan/.mtdata/mtdata.index.0.3.5.pkl",
    )
    parser.add_argument("--dry-run", help="Dry run", default=False, action="store_true")
    return parser.parse_args()


def main(args):
    custom_dataset_dir = args.data_dir
    if os.path.isfile(custom_dataset_dir):
        with open(custom_dataset_dir, "rb") as f:
            df = pickle.load(f)
        # df_groups = df.groupby("qanta_id")
        aa = 0


if __name__ == "__main__":
    args = parse_args()
    main(args)
