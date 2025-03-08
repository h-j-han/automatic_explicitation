from typing import List
import logging, sys, os

logger = logging.getLogger(__name__)


def precision2(actual: List[int], predicted: List[int], print_all=True):
    # No need to get True Negative
    # No need to consider the number of total instance
    # only tp fp fn matters!
    # Initialize the true positive, true negative, and false positive lists
    tp = []
    # tn = []
    fp = []
    fn = []
    merged = list(set(actual + predicted))
    # Iterate over the actual and predicted lists and populate the true positive,
    # true negative, and false positive lists accordingly
    for i in merged:
        if i in actual and i in predicted:
            tp.append(i)
        elif i not in actual and i in predicted:
            fp.append(i)
        elif i in actual and i not in predicted:
            fn.append(i)
        # elif i not in actual and i not in predicted:
        #     tn.append(i)
    # Calculate precision, recall, and F1 score
    precision = len(tp) / (len(tp) + len(fp))
    if len(actual) > 0:
        recall = len(tp) / len(actual)  # Actual = true positive + false negative
    else:
        recall = 0
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0

    # Print the results
    if print_all:
        logger.info(f"{len(actual)=}, {len(predicted)=}")
        logger.info(f"True positives({len(tp)=}): {tp}")
        # logger.info(f"True negatives: {tn}")
        logger.info(f"False positives({len(fp)=}): {fp}")
        logger.info(f"False negatives({len(fn)=}): {fn}")
        logger.info(f"Precision: {precision:.3f}")
        logger.info(f"Recall: {recall:.3f}")
        logger.info(f"F1 score: {f1:.3f}")
    return {
        "tp": tp,
        # "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def precision(actual: List[int], predicted: List[int], print_all=True):
    # No need to get True Negative
    # No need to consider the number of total instance
    # only tp fp fn matters!
    # Initialize the true positive, true negative, and false positive lists
    tp = []
    # tn = []
    fp = []
    fn = []
    # Iterate over the actual and predicted lists and populate the true positive,
    # true negative, and false positive lists accordingly
    for i in range(max(max(actual), max(predicted)) + 1):
        if i in actual and i in predicted:
            tp.append(i)
        elif i not in actual and i in predicted:
            fp.append(i)
        elif i in actual and i not in predicted:
            fn.append(i)
        # elif i not in actual and i not in predicted:
        #     tn.append(i)
    # Calculate precision, recall, and F1 score
    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / len(actual)  # Actual = true positive + false negative
    f1 = 2 * (precision * recall) / (precision + recall)

    # Print the results
    if print_all:
        logger.info(f"{len(actual)=}, {len(predicted)=}")
        logger.info(f"True positives({len(tp)=}): {tp}")
        # logger.info(f"True negatives: {tn}")
        logger.info(f"False positives({len(fp)=}): {fp}")
        logger.info(f"False negatives({len(fn)=}): {fn}")
        logger.info(f"Precision: {precision:.3f}")
        logger.info(f"Recall: {recall:.3f}")
        logger.info(f"F1 score: {f1:.3f}")
    return {
        "tp": tp,
        # "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    logging.basicConfig(
        stream=sys.stdout,
        level=os.environ.get("LOGLEVEL", "INFO").upper(),
        format="%(asctime)s %(message)s",
    )
    # fmt: off
    idx_list=[13, 29, 76, 101, 104, 148, 203, 265, 299, 322, 328, 333, 347, 358, 397, 421, 444, 467, 479, 586, 637, 746, 758, 775, 829, 831, 858, 881, 897, 902, 974]
    wiki1000_ans = [13, 34, 76, 101, 148, 203, 104, 265, 328, 397, 586, 746, 829, 858, 881]
    # fmt: on
    dict_out = precision(wiki1000_ans, idx_list)
