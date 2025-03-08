import numpy as np
import math
import re
import json
from typing import List, Dict, Iterable, Optional, Tuple, NamedTuple
import os
import pickle
import numpy as np


def answer_matched(gus: str, ans) -> bool:
    if isinstance(ans, str) and gus.lower().replace(" ", "").replace("_", "").replace(
        "-", ""
    ) == ans.lower().replace(" ", "").replace("_", "").replace("-", ""):
        return True
    # src is raw, unprocessed answer text
    # english is processed, and used as answer, so same matching algorithm
    # {'pl': 'DŁUGOŚĆ GEOGRAFICZNA', 'en': 'Longitude'}
    if isinstance(ans, dict):
        for l in ans:
            if l == "en":
                if gus.lower().replace(" ", "").replace("_", "").replace(
                    "-", ""
                ) == ans[l].lower().replace(" ", "").replace("_", "").replace("-", ""):
                    return True
            else:
                if gus.lower() in ans[l].lower():
                    return True
    return False


def get_mrr_steps(guesses, true_ans):
    rs = [
        np.asarray(
            [answer_matched(gus, true_ans) for gus in step["guesses"]]
        ).nonzero()[0]
        for step in guesses
    ]
    return [1.0 / (r[0] + 1) if r.size else 0.0 for r in rs]


def get_log_mrr_steps(mrrs):
    return [math.log(mrr) if mrr != 0 else -4 for mrr in mrrs]


def get_mrr_step(guess, true_ans):
    r = np.asarray(
        [answer_matched(gus, true_ans) for gus in guess["guesses"]]
    ).nonzero()[0]
    return 1.0 / (r[0] + 1) if r.size else 0.0


def qidmap_lqus2dict(questions, keytype="int"):
    qidmap = {}
    for qus in questions:
        qanta_id = qus["qanta_id"]
        if keytype == "int":
            qanta_id = int(qanta_id)
        qidmap[qanta_id] = qus
    return qidmap


def load_qanta_json2dict(file_path: str) -> Dict:
    orig = {}
    with open(file_path, "rb") as f:
        tmp = json.load(f)
        for q in tmp["questions"]:
            orig[q["qanta_id"]] = q
    return orig


def words_update_text(text, word_skip: int) -> Tuple[List[str], List[int]]:
    """
    Returns runs of the question based on skipping word_skip characters at a time. Also returns the indices used

    q: name this first united states president.
    runs with word_skip=2:
    ['name this',
        'name this first united',
        'name this first united state president.']

    :param word_skip: Number of words to skip each time
    """
    query_part = " "
    char_indices = [match.start() for match in re.finditer(query_part, text)]
    char_indices = char_indices[word_skip - 1 :: word_skip]
    char_indices.append(len(text))
    return [text[:i] for i in char_indices], char_indices


def words_entity_skip(text, word_skip: int, new_tllents_char_idx: List[List[int]]):
    lsubtexts = []
    lindices = []
    prevlen = 0
    for lents in new_tllents_char_idx:
        sci = lents[0]
        eci = lents[1]
        subtexts, indices = words_update_text(text[:sci], word_skip)
        for subtext, indice in zip(subtexts, indices):
            if len(subtext) > prevlen:
                lsubtexts.append(subtext)
                lindices.append(indice)
        # Dangerous assumption that eci is not the last char in the ques text
        lsubtexts.append(text[: eci + 1])
        lindices.append(eci + 1)
        prevlen = len(lsubtexts[-1])
    subtexts, indices = words_update_text(text, word_skip)
    for subtext, indice in zip(subtexts, indices):
        if len(subtext) > prevlen:
            lsubtexts.append(subtext)
            lindices.append(indice)
    return lsubtexts, lindices


def char_entity_skip(text, skip: int, new_tllents_char_idx: List[List[int]]):
    tokenizations = sorted(new_tllents_char_idx)
    last = len(text)
    start = 0
    result = []
    for pair in tokenizations:
        if pair[0] == 0:
            result.append(pair[0])
        while start < pair[0]:
            if start > 0:
                result.append(start)
            start += skip
        if pair[0] not in result:
            result.append(pair[0])
        start = pair[1] + 1
    while start < last:  # Add indices after the last pair until we reach the last skip
        result.append(start)
        start += skip
    result.append(last)
    return [text[:i] for i in result], result


def decide_char_skip(orig_src_text, orig_sllents_char_idx, char_skip, max_batch_size):
    orig_lsubtexts, orig_lindices = char_entity_skip(
        orig_src_text, char_skip, orig_sllents_char_idx
    )
    assert len(set(orig_lindices)) == len(orig_lindices)
    for si, ei in orig_sllents_char_idx:
        assert si in orig_lindices
        assert ei + 1 in orig_lindices
    orig_new_char_skip = char_skip
    while len(orig_lindices) > max_batch_size:
        orig_new_char_skip = orig_new_char_skip + 1
        orig_lsubtexts, orig_lindices = char_entity_skip(
            orig_src_text, orig_new_char_skip, orig_sllents_char_idx
        )
        assert len(set(orig_lindices)) == len(orig_lindices)
        for si, ei in orig_sllents_char_idx:
            assert si in orig_lindices
            assert ei + 1 in orig_lindices
    return orig_new_char_skip


class CurveScore:
    def __init__(self, report_dir="output/reporting"):
        ckp_dir = os.path.join(report_dir, "curve_pipeline.pkl")
        if os.path.isfile(ckp_dir):
            print("loading pipeline")
            with open(ckp_dir, "rb") as f:
                self.pipeline = pickle.load(f)
        else:
            print("fitting pipeline")
            self.pipeline = None

    def get_weight(self, x):
        if self.pipeline is None:
            return self.winning_players_proportion(x)
        return self.pipeline.predict(np.asarray([[x]]))[0]


    def winning_players_proportion(t: float):
        """Proportion of players that have correctly guessed the answer by length t.add()
        https://github.com/Pinafore/848-hw/blob/main/hw1/run_e2e_eval.py
        This metric is implemented in https://arxiv.org/pdf/1904.04792.pdf (Page 26)."""
        return min(1.0, 0.9931 + 0.0775 * t - 1.278 * t ** 2 + 0.588 * t ** 3)


    def trans_score2(
        self, guesses, question, idx_info_dict, threshold=0.5, newans=None
    ):
        # Use of answer_matched is the diff of ~2 vs ~
        """guesses is a list of {'guess': GUESS, 'buzz': True/False}"""
        answer = question["page"] if newans is None else newans
        char_length = len(question["text"])
        src_char_length = len(idx_info_dict["src"])
        if "buzz" in guesses[0]:
            buzzes = [x["buzz"] for x in guesses]
        else:
            buzzes = [x["score"] > threshold for x in guesses]
        if True not in buzzes:
            return 0
        buzz_index = buzzes.index(True)
        char_index = guesses[buzz_index]["char_index"]
        src_char_index = idx_info_dict["srccharidx"][
            idx_info_dict["tgtdelay"][buzz_index]
        ]
        rel_position = (1.0 * src_char_index) / src_char_length
        weight = self.get_weight(rel_position)
        result = answer_matched(guesses[buzz_index]["guess"], answer)
        return weight * result

    def trans_score_optimal2(self, guesses, question, idx_info_dict, newans=None):
        # Assumption : word white space split
        # Use of answer_matched is the diff of ~2 vs ~
        """score with an optimal buzzer"""
        answer = question["page"] if newans is None else newans
        char_length = len(question["text"])
        buzz_index = char_length
        src_char_length = len(idx_info_dict["src"])
        src_char_index = src_char_length
        for i, g in enumerate(guesses):
            if answer_matched(g["guess"], answer):
                char_index = g["char_index"]
                buzz_index = len(question["text"][: g["char_index"]].split()) - 1
                # if buzz_index != i:
                # print(f'{i=} is not same with {buzz_index=}')
                src_char_index = idx_info_dict["srccharidx"][
                    idx_info_dict["tgtdelay"][buzz_index]
                ]
                break
        rel_position = (1.0 * src_char_index) / src_char_length
        return self.get_weight(rel_position)

    def trans_score_optimal2_exp(self, guesses, question, idx_info_dict, newans=None):
        # No white space assumption. This is for explicitation split
        # Use of answer_matched is the diff of ~2 vs ~
        """score with an optimal buzzer"""
        answer = question["page"] if newans is None else newans
        char_length = len(question["text"])
        buzz_index = char_length
        src_char_length = len(idx_info_dict["src"])
        src_char_index = src_char_length
        for i, g in enumerate(guesses):
            if answer_matched(g["guess"], answer):
                char_index = g["char_index"]
                src_char_index = idx_info_dict["srccharidx"][i]
                break
        rel_position = (1.0 * src_char_index) / src_char_length
        return self.get_weight(rel_position)

    def trans_score(self, guesses, question, idx_info_dict):
        """guesses is a list of {'guess': GUESS, 'buzz': True/False}"""
        char_length = len(question["text"])
        src_char_length = len(idx_info_dict["src"])
        buzzes = [x["buzz"] for x in guesses]
        if True not in buzzes:
            return 0
        buzz_index = buzzes.index(True)
        char_index = guesses[buzz_index]["char_index"]
        src_char_index = idx_info_dict["srccharidx"][
            idx_info_dict["tgtdelay"][buzz_index]
        ]
        rel_position = (1.0 * src_char_index) / src_char_length
        weight = self.get_weight(rel_position)
        result = guesses[buzz_index]["guess"] == question["page"]
        return weight * result

    def trans_score_optimal(self, guesses, question, idx_info_dict):
        """score with an optimal buzzer"""
        char_length = len(question["text"])
        buzz_index = char_length
        src_char_length = len(idx_info_dict["src"])
        src_char_index = src_char_length
        for i, g in enumerate(guesses):
            if g["guess"] == question["page"]:
                char_index = g["char_index"]
                buzz_index = len(question["text"][: g["char_index"]].split()) - 1
                # if buzz_index != i:
                # print(f'{i=} is not same with {buzz_index=}')
                src_char_index = idx_info_dict["srccharidx"][
                    idx_info_dict["tgtdelay"][buzz_index]
                ]
                break
        rel_position = (1.0 * src_char_index) / src_char_length
        return self.get_weight(rel_position)

    def score_orig(self, guesses, question, threshold=0.5, newans=None):
        """guesses is a list of {'guess': GUESS, 'buzz': True/False}"""
        answer = question["page"] if newans is None else newans
        char_length = len(question["text"])
        if "buzz" in guesses[0]:
            buzzes = [x["buzz"] for x in guesses]
        else:
            buzzes = [x["score"] > threshold for x in guesses]
        if True not in buzzes:
            return 0
        buzz_index = buzzes.index(True)
        rel_position = (1.0 * guesses[buzz_index]["char_index"]) / char_length
        weight = self.get_weight(rel_position)
        # result = guesses[buzz_index]["guess"] == question["page"]
        result = answer_matched(guesses[buzz_index]["guess"], answer)
        return weight * result

    def score(self, guesses, question):
        """guesses is a list of {'guess': GUESS, 'buzz': True/False}"""
        char_length = len(question["text"])
        buzzes = [x["buzz"] for x in guesses]
        if True not in buzzes:
            return 0, 0, None, -1
        buzz_index = buzzes.index(True)
        char_index = guesses[buzz_index]["char_index"]
        rel_position = (1.0 * guesses[buzz_index]["char_index"]) / char_length
        weight = self.get_weight(rel_position)
        result = guesses[buzz_index]["guess"] == question["page"]
        return weight * result, rel_position, guesses[buzz_index]["guess"], char_index

    def score_optimal(self, guesses, question, newans=None):
        """score with an optimal buzzer"""
        answer = question["page"] if newans is None else newans
        char_length = len(question["text"])
        buzz_index = char_length
        for g in guesses:
            if answer_matched(g["guess"], answer):
                buzz_index = g["char_index"]
                break
        rel_position = (1.0 * buzz_index) / char_length
        return self.get_weight(rel_position)

    def accuracy2(self, guesses, question, newans=None):
        """score with an accuracy not considering buzzer"""
        answer = question["page"] if newans is None else newans
        guess = guesses[-1]["guess"]
        return answer_matched(guess, answer)

    def mrr_full2(self, guesses, question, newans=None):
        """https://en.wikipedia.org/wiki/Mean_reciprocal_rank"""
        answer = question["page"] if newans is None else newans
        list_guess = guesses[-1]["guesses"]
        list_corr = [answer_matched(guess, answer) for guess in list_guess]
        r = np.asarray(list_corr).nonzero()[0]
        return 1.0 / (r[0] + 1) if r.size else 0.0

    def accuracy(self, guesses, question):
        """score with an accuracy not considering buzzer"""
        guess = guesses[-1]["guess"]
        return guess == question["page"]

    def mrr_full(self, guesses, question):
        """https://en.wikipedia.org/wiki/Mean_reciprocal_rank"""
        list_guess = guesses[-1]["guesses"]
        list_corr = [guess == question["page"] for guess in list_guess]
        r = np.asarray(list_corr).nonzero()[0]
        return 1.0 / (r[0] + 1) if r.size else 0.0

    def buzzed(self, guesses, question):
        """buzzed or not"""
        buzzes = [x["buzz"] for x in guesses]
        return True not in buzzes

    def trans_buzz_position_nonbuzz_one(self, guesses, question, idx_info_dict):
        char_length = len(question["text"])
        src_char_length = len(idx_info_dict["src"])
        buzzes = [x["buzz"] for x in guesses]
        if True not in buzzes:
            return 1
        buzz_index = buzzes.index(True)
        char_index = guesses[buzz_index]["char_index"]
        src_char_index = idx_info_dict["srccharidx"][
            idx_info_dict["tgtdelay"][buzz_index]
        ]
        rel_position = (1.0 * src_char_index) / src_char_length
        return rel_position

    def trans_buzz_position_nonbuzz_minusone(self, guesses, question, idx_info_dict):
        char_length = len(question["text"])
        src_char_length = len(idx_info_dict["src"])
        buzzes = [x["buzz"] for x in guesses]
        if True not in buzzes:
            return -1, -1, -1
        buzz_index = buzzes.index(True)
        char_index = guesses[buzz_index]["char_index"]
        src_char_index = idx_info_dict["srccharidx"][
            idx_info_dict["tgtdelay"][buzz_index]
        ]
        rel_position = (1.0 * src_char_index) / src_char_length
        return rel_position, char_index, src_char_index

    def score_stable(self, guesses, question):
        """score with an optimal buzzer"""
        char_length = len(question["text"])
        buzz_index = char_length
        for g in guesses[::-1]:
            if g["guess"] != question["page"]:
                buzz_index = g["char_index"]
                break
        rel_position = (1.0 * buzz_index) / char_length
        return self.get_weight(rel_position)
