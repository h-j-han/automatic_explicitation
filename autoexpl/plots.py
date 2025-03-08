import sys
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from autoexpl.tools.precision import precision2

HORIZONTAL_FULL_FIG_SIZE = 18
VERTICAL_FIG_SIZE = 3

NAMECONV = {
    "es": "Spanish QA task",
    "pl": "Polish QA task",
    "en": "English QA task",
    "plqbv1ht512": "XQB-pl",
    "esqbv1htall": "XQB-es",
    "instanceof": "Short",
    "wikides": "Mid",
    "wikipara": "Long",
    "ew": "Expected Win (EW)",
    "ewo": "Expected Win Oracle  (EWO)",
    "full_acc": "Full Input Accuracy",
    "mrr_eci": "MRR after the entity",
    "mrr_avg": "Avg MRR",
    "whole": "All Entities",
    "relatedQ": "Related Qs",
    "~relatedQ": "Not Related Qs",
    "relatedE": "Explicitation",
    "~relatedE": "Non-explicitation Entites",
    "AincludedX": "Answer Containing Expl",
    "~AincludedX": "Not Answer Containing Expl",
    "relatedQrelatedE": "Explicitation\nAmong Related Qs",
    "relatedQ~relatedE": "Not Detected Entites\nAmong Related Qs",
    "relatedQAincludedX": "Answer Containing Expl\nAmong Related Qs",
    "relatedQ~AincludedX": "Not Answer Containing Expl\nAmong Related Qs",
    "relatedEAincludedX": "Answer Containing Expl\nAmong Related Es",
    "relatedE~AincludedX": "Not Answer Containing Expl\nAmong Related Es",
}


def plot8(plot_func, llcompare, data, *args, **kwargs):
    nrow = 2
    ncol = 4
    hor_figsize = HORIZONTAL_FULL_FIG_SIZE
    fig, axes = plt.subplots(2, 4, figsize=(hor_figsize, VERTICAL_FIG_SIZE * nrow))
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.9, wspace=0.5)  # wspace=0.2,
    for i, lcompare in enumerate(llcompare):
        ax = axes[int(i / ncol)][int(i % ncol)]
        plot_func(ax, lcompare, data, *args, **kwargs)
    return plt


def plot6(plot_func, llcompare, data, *args, **kwargs):
    nrow = 2
    ncol = 3
    hor_figsize = HORIZONTAL_FULL_FIG_SIZE
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(hor_figsize, VERTICAL_FIG_SIZE * nrow)
    )
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # wspace=0.2,
    for i, lcompare in enumerate(llcompare):
        ax = axes[int(i / ncol)][int(i % ncol)]
        plot_func(ax, lcompare, data, *args, **kwargs)
    return plt


def plot3(plot_func, llcompare, data, largs, **kwargs):
    nrow = 1
    hor_figsize = HORIZONTAL_FULL_FIG_SIZE
    fig, axes = plt.subplots(1, 3, figsize=(hor_figsize, VERTICAL_FIG_SIZE * nrow))
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # wspace=0.2,
    for i, lcompare in enumerate(llcompare):
        ax = axes[i]
        args = largs[i]
        plot_func(ax, lcompare, data, **args, **kwargs)
    return fig


def plot2(plot_func, llcompare, data, *args, **kwargs):
    nrow = 1
    ncol = 2
    hor_figsize = HORIZONTAL_FULL_FIG_SIZE
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(hor_figsize, VERTICAL_FIG_SIZE * nrow + 1)
    )
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # wspace=0.2,
    for i, lcompare in enumerate(llcompare):
        ax = axes[i]
        plot_func(ax, lcompare, data, *args, **kwargs)
    return plt


def plot22(plot_func, llcompare, data, *args, **kwargs):
    nrow = 2
    ncol = 3
    hor_figsize = HORIZONTAL_FULL_FIG_SIZE
    fig, axes = plt.subplots(
        nrow, ncol, figsize=(hor_figsize, VERTICAL_FIG_SIZE * nrow)
    )
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)  # wspace=0.2,
    for i, lcompare in enumerate(llcompare):
        ax = axes[int(i / ncol)][int(i % ncol)]
        plot_func(ax, lcompare, data, *args, **kwargs)
    return plt


def plot4(plot_func, llcompare, data, *args, **kwargs):
    nrow = 1
    hor_figsize = HORIZONTAL_FULL_FIG_SIZE
    fig, axes = plt.subplots(1, 4, figsize=(hor_figsize, VERTICAL_FIG_SIZE * nrow))
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.5)  # wspace=0.2,
    for i, lcompare in enumerate(llcompare):
        ax = axes[i]
        plot_func(ax, lcompare, data, *args, **kwargs)
    return plt


def plot_binning(
    ax,
    lcompare,
    dfs,
    l_ent_id_related,
    qid_filter=None,
    loc="lower right",
    ylimax=-1,
    nationality="Polish related",
    title_prefix=None,
):
    lthr, l_n_eid_qa, l_n_id_ours_in_qa = binning(
        lcompare, dfs, l_ent_id_related, qid_filter=qid_filter
    )
    lthr_string = [
        "(" + f"{tup[0]:.2f}" + ",\n" + f"{tup[1]:.2f}" + "]" for tup in lthr
    ]
    llabel = ["Partial" for i in range(len(lthr))]
    llabel[0] = "Full Range"
    d = {
        "tau": lthr_string,
        "total": l_n_eid_qa,
        "related": l_n_id_ours_in_qa,
        "label": llabel,
    }
    if ylimax > 0:
        ax.set(ylim=(0.0, ylimax))
    # sns.set_color_codes("pastel")
    # sns.set_color_codes("muted")
    ax = sns.barplot(
        x="tau",
        y="total",
        data=d,
        capsize=0.3,
        dodge=False,
        ax=ax,
        zorder=1,
        # color="b",
        hue="label",
        palette="pastel",
    )
    # print(d)

    ax = sns.barplot(
        x="tau",
        y="related",
        data=d,
        capsize=0.3,
        dodge=False,
        ax=ax,
        zorder=1,
        color="r",
        hue="label",
        palette="bright",
    )
    ax.grid(True)
    leg_handles = ax.get_legend_handles_labels()
    leg_handles[1][2] = nationality
    leg_handles[1][3] = nationality
    ax.legend(leg_handles[0], leg_handles[1])
    ax.set_title(f"{title_prefix}{lcompare[1]} - {lcompare[0]}" + r" $\geq \tau$")
    ax.set_xlabel(r"Range of $\tau$")
    ax.set_ylabel("Number of Entities in each bin")
    for i, p in enumerate(ax.patches):
        if i == 12:
            ax.annotate(
                f"{100*l_n_id_ours_in_qa[0]/l_n_eid_qa[0]:.1f}%",
                (p.get_x() - 0.03, p.get_height() + ax.get_ylim()[1] * 0.11),
            )
            ax.annotate(
                f"# {l_n_id_ours_in_qa[0]}",
                (p.get_x() - 0.03, p.get_height() + ax.get_ylim()[1] * 0.02),
            )
        if i > 18:
            ax.annotate(
                f"# {l_n_id_ours_in_qa[i - 18]}",
                (p.get_x() - 0.03, p.get_height() + ax.get_ylim()[1] * 0.02),
            )
    return ax


def plot_binning_among_related_list(
    ax,
    lcompare,
    dfs,
    l_ent_id_related,
    qid_filter=None,
    loc="lower right",
    ylimax=-1,
    nationality="Polish related",
    title_prefix=None,
):
    lthr, l_n_eid_qa, l_n_id_ours_in_qa = binning(
        lcompare, dfs, l_ent_id_related, qid_filter=qid_filter
    )
    lthr_string = [
        "(" + f"{tup[0]:.2f}" + ",\n" + f"{tup[1]:.2f}" + "]" for tup in lthr
    ]
    llabel = ["Partial" for i in range(len(lthr))]
    llabel[0] = "Full Range"
    d = {
        "tau": lthr_string,
        "total": [len(l_ent_id_related) for _ in range(len(lthr))],
        "related": l_n_id_ours_in_qa,
        "label": llabel,
    }
    if ylimax > 0:
        ax.set(ylim=(0.0, ylimax))
    # sns.set_color_codes("pastel")
    # sns.set_color_codes("muted")
    ax = sns.barplot(
        x="tau",
        y="total",
        data=d,
        capsize=0.3,
        dodge=False,
        ax=ax,
        zorder=1,
        # color="b",
        hue="label",
        palette="pastel",
    )
    # print(d)

    ax = sns.barplot(
        x="tau",
        y="related",
        data=d,
        capsize=0.3,
        dodge=False,
        ax=ax,
        zorder=1,
        color="r",
        hue="label",
        palette="bright",
    )
    ax.grid(True)
    leg_handles = ax.get_legend_handles_labels()
    leg_handles[1][0] = nationality
    leg_handles[1][1] = nationality
    ax.legend(leg_handles[0], leg_handles[1])
    ax.set_title(f"{title_prefix}{lcompare[1]} - {lcompare[0]}" + r" $\geq \tau$")
    ax.set_xlabel(r"Range of $\tau$")
    ax.set_ylabel("Number of Entities in each bin")
    for i, p in enumerate(ax.patches):
        if i == 12:  # first column partial bar
            ax.annotate(
                f"{100*l_n_id_ours_in_qa[0]/len(l_ent_id_related):.1f}%",
                (p.get_x() - 0.03, p.get_height() + ax.get_ylim()[1] * 0.11),
            )
            ax.annotate(
                f"# {l_n_id_ours_in_qa[0]}",
                (p.get_x() - 0.03, p.get_height() + ax.get_ylim()[1] * 0.02),
            )
        if i > 18:
            ax.annotate(
                f"# {l_n_id_ours_in_qa[i - 18]}",
                (p.get_x() - 0.03, p.get_height() + ax.get_ylim()[1] * 0.02),
            )
    return ax


def inclusion_cond(df, dcond):  # , vkey2, list2, neg2):
    vkey = dcond["vkey"]
    neg = dcond["neg"]
    clist = dcond["clist"]
    if not neg:
        cond = df[vkey].isin(clist)
    else:
        cond = ~df[vkey].isin(clist)
    return cond


def matching_cond(df, dcond):  # , vkey2, list2, neg2):
    vkey = dcond["vkey"]
    val = dcond["val"]
    cond = df[vkey] == val
    return cond


def dfs_with_multi_conds(df, ld_conds):
    if len(ld_conds) == 1:
        return df[inclusion_cond(df, ld_conds[0])]
    elif len(ld_conds) == 2:
        return df[inclusion_cond(df, ld_conds[0]) & inclusion_cond(df, ld_conds[1])]
    elif len(ld_conds) == 2:
        return df[
            inclusion_cond(df, ld_conds[0])
            & inclusion_cond(df, ld_conds[1])
            & inclusion_cond(df, ld_conds[2])
        ]
    else:
        raise NotImplementedError


def conditiondfs(
    condname,
    l_orig_qid_whole=None,
    l_orig_qid_related=None,
    l_ent_id_related=None,
    l_exp_answer_incuded=None,
):
    if condname == "whole":
        ldcond = [{"vkey": "orig_qid", "clist": l_orig_qid_whole, "neg": False}]
    elif condname == "relatedQ":
        ldcond = [{"vkey": "orig_qid", "clist": l_orig_qid_related, "neg": False}]
    elif condname == "~relatedQ":
        ldcond = [{"vkey": "orig_qid", "clist": l_orig_qid_related, "neg": True}]
    elif condname == "relatedE":
        ldcond = [{"vkey": "ent_id", "clist": l_ent_id_related, "neg": False}]
    elif condname == "~relatedE":
        ldcond = [{"vkey": "ent_id", "clist": l_ent_id_related, "neg": True}]
    elif condname == "AincludedX":
        ldcond = [{"vkey": "exp_qid", "clist": l_exp_answer_incuded, "neg": False}]
    elif condname == "~AincludedX":
        ldcond = [{"vkey": "exp_qid", "clist": l_exp_answer_incuded, "neg": True}]
    elif condname == "relatedQrelatedE":
        ldcond = [
            {"vkey": "orig_qid", "clist": l_orig_qid_related, "neg": False},
            {"vkey": "ent_id", "clist": l_ent_id_related, "neg": False},
        ]
    elif condname == "relatedQ~relatedE":
        ldcond = [
            {"vkey": "orig_qid", "clist": l_orig_qid_related, "neg": False},
            {"vkey": "ent_id", "clist": l_ent_id_related, "neg": True},
        ]
    elif condname == "relatedQAincludedX":
        ldcond = [
            {"vkey": "orig_qid", "clist": l_orig_qid_related, "neg": False},
            {"vkey": "exp_qid", "clist": l_exp_answer_incuded, "neg": False},
        ]
    elif condname == "relatedQ~AincludedX":
        ldcond = [
            {"vkey": "orig_qid", "clist": l_orig_qid_related, "neg": False},
            {"vkey": "exp_qid", "clist": l_exp_answer_incuded, "neg": True},
        ]
    elif condname == "relatedEAincludedX":
        ldcond = [
            {"vkey": "ent_id", "clist": l_ent_id_related, "neg": False},
            {"vkey": "exp_qid", "clist": l_exp_answer_incuded, "neg": False},
        ]
    elif condname == "relatedE~AincludedX":
        ldcond = [
            {"vkey": "ent_id", "clist": l_ent_id_related, "neg": False},
            {"vkey": "exp_qid", "clist": l_exp_answer_incuded, "neg": True},
        ]
    else:
        print(condname)
        raise NotImplementedError
    return ldcond


# [["condtype" : 'whole', "lcompare":["orig_ew", "exp_tew"], ]
# ['relatedQ', ["orig_ew", "exp_tew"]]
# ['~relatedQ', ["orig_ew", "exp_tew"]]
# ['relatedE', ["orig_ew", "exp_tew"]]
# ['~relatedE', ["orig_ew", "exp_tew"]]
# ['relatedQrelatedE', ["orig_ew", "exp_tew"]]
# ['relatedQ~relatedE', ["orig_ew", "exp_tew"]]]

# [['whole', ["orig_ew", "exp_tew"]]
# ['whole', ["orig_ewo", "exp_tewo"]]
# ['whole', ["orig_full_acc", "exp_full_acc"]]
# ['whole', ["orig_mrr_eci", "exp_mrr_eci"]]
# ['whole', ["orig_mrr_end", "exp_mrr_end"]]
# ['whole', ["orig_mrr_avg", "exp_mrr_avg"]]
# ]


def average_list(dfs, condtype, lcompare, **dlist):
    ldcond = conditiondfs(condtype, **dlist)
    cdfs = dfs_with_multi_conds(dfs, ldcond)
    v1 = np.mean(cdfs[lcompare[0]])  #
    v2 = np.mean(cdfs[lcompare[1]])  #
    increate_rate = (v2 - v1) / v1
    return v1, v2, increate_rate * 100


def avgllist_gentype(dfs, condtype, lcompare, gentype, **dlist):
    ldcond = conditiondfs(condtype, **dlist)
    cdfs = dfs_with_multi_conds(dfs, ldcond)
    # if gentype == "instanceof":
    #     gcdfs = cdfs[(cdfs["gentype"] == gentype) | (cdfs["gentype"] == "countryof")]
    # else:
    #     gcdfs = cdfs[cdfs["gentype"] == gentype]
    gcdfs = cdfs[cdfs["gentype"] == gentype]
    v1 = np.mean(gcdfs[lcompare[0]])  #
    v2 = np.mean(gcdfs[lcompare[1]])  #
    increate_rate = (v2 - v1) / v1
    return v1, v2, increate_rate * 100


def avgllist_gentype_ansinc(dfs, condtype, lcompare, gentype, lang="en", **dlist):
    ldcond = conditiondfs(condtype, **dlist)
    cdfs = dfs_with_multi_conds(dfs, ldcond)
    # if gentype == "instanceof":
    #     gcdfs = cdfs[(cdfs["gentype"] == gentype) | (cdfs["gentype"] == "countryof")]
    # else:
    #     gcdfs = cdfs[cdfs["gentype"] == gentype]
    gcdfs = cdfs[cdfs["gentype"] == gentype]
    igcdfs = gcdfs[gcdfs["exp_qid"].isin(dlist["l_exp_answer_incuded"])]
    # igcdfs = gcdfs[gcdfs["exp_qid"].isin(dlist["l_exp_answer_incuded"][lang])]
    v1 = np.mean(gcdfs[lcompare[0]])  #
    v2 = np.mean(gcdfs[lcompare[1]])  #
    increate_rate = (v2 - v1) / v1
    if len(gcdfs) > 0:
        ansincl_rate = len(igcdfs) / len(gcdfs)
    else:
        ansincl_rate = 0
    # print(f"{v1=} {v2=} {increate_rate=}")
    return v1, v2, increate_rate * 100, ansincl_rate * 100


def plot_compare_bar_pair(
    ax,
    lcomb,
    dfs,
    title_prefix=None,
    y1maxlim=0.75,
    y2maxlim=250,
    loc="lower right",
    turnofflegend=False,
    man_ylabel=None,
    all_explicitation=False,
    **dlist,
):
    lorig = []
    lexp = []
    lrate = []
    lcondtype = []
    lcompval = []
    lgentype = []
    lansincl = []
    llang = []
    for comb in lcomb:
        # print(comb)
        condtype = comb["condtype"]
        lcompare = comb["lcompare"]
        gentype = comb["gentype"]
        lang = comb["lang"]
        ov, ev, ir, ar = avgllist_gentype_ansinc(
            dfs[dfs["lang"] == lang], condtype, lcompare, gentype, lang=lang, **dlist
        )
        lorig.append(ov)
        lexp.append(ev)
        lrate.append(ir)
        llang.append(lang)
        lansincl.append(ar)
        lcondtype.append(condtype)
        lcompval.append(lcompare[0].replace("orig_", ""))
        lgentype.append(gentype)
        # print(lcondtype)
        # print(lcompval)

    llabel = llang
    ylabel = lcondtype[0] if man_ylabel is None else man_ylabel
    commonval = lcompval[0]
    # ax.set_title(f"{lcondtype[0]}")
    # y2maxlim = (int(max(lrate)/10) + 1)*10
    # y1maxlim = max(max(lorig), max(lexp))

    # print(llabel)
    y1minlim = ax.get_ylim()[0]
    ax.grid(True)

    ax.set(ylim=(y1minlim, y1maxlim))
    ax2 = ax.twinx()
    # y2minlim = max(0, min(lrate))
    y2minlim = 0
    ax2.set(ylim=(y2minlim, y2maxlim))
    data = []
    for i, (
        orig,
        exp,
        rate,
        aninr,
        condtype,
        compval,
        lang,
        gentype,
        label,
    ) in enumerate(
        zip(lorig, lexp, lrate, lansincl, lcondtype, lcompval, llang, lgentype, llabel)
    ):
        data.append(
            {
                "val": orig,
                "class": "Original",
                "rate": rate,
                "lang": lang,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
        data.append(
            {
                "val": exp,
                "class": (
                    f"Additional info"  # ({NAMECONV[gentype]})"
                    if not all_explicitation
                    else "Explicitation"
                ),
                "rate": rate,
                "lang": lang,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
        # print(rate)
        data.append(
            {
                "val": rate * (y1maxlim - y1minlim) / (y2maxlim - y2minlim),
                "class": "Increase Rate",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
        # data.append(
        #     {
        #         "val": aninr * (y1maxlim - y1minlim) / (y2maxlim - y2minlim),
        #         "class": "Answer Inclusion Rate",
        #         "rate": rate,
        #         "lang": lang,
        #         "condtype": condtype,
        #         "compval": compval,
        #         "label": NAMECONV[label],
        #     }
        # )
    # print(data)
    d = pd.DataFrame(data)
    ax = sns.barplot(
        x="label",
        y="val",
        data=d,  # [(d["class"] == "orig") | (d["class"] == "exp")],
        capsize=0.3,
        # dodge=False,
        ax=ax,
        zorder=1,
        # color="b",
        hue="class",
        palette="pastel",
    )
    if "lower center" in loc:
        ax.legend(
            title="",
            loc=loc,
            bbox_to_anchor=(0.5, -0.65),
        )  # to remove legend title
    else:
        ax.legend(title="", loc=loc)
    ax.set_title(f"{title_prefix}{NAMECONV[ylabel]}")
    ax.set_xlabel("")  # xlabel
    ax.set_ylabel(NAMECONV[commonval])
    ax2.set_ylabel("Increase Rate(%)")
    # if ylabel == "ewo" or turnofflegend:
    #     ax.legend([], [], frameon=False)  # turnoff legend
    for i, p in enumerate(ax.patches):
        if i > 3:
            ax.annotate(
                f"{p.get_height() * (y2maxlim - y2minlim) / (y1maxlim - y1minlim):.1f}%",
                (p.get_x(), p.get_height() + ax.get_ylim()[1] * 0.03),
            )
        else:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() + 0.03, p.get_height() + ax.get_ylim()[1] * 0.03),
                # (p.get_x() , p.get_height() + ax.get_ylim()[1] * 0.03),
            )


def plot_compare_bar_gentype_ansincl(
    ax,
    lcomb,
    dfs,
    title_prefix=None,
    y1maxlim=0.75,
    y2maxlim=250,
    loc="lower right",
    turnofflegend=False,
    **dlist,
):
    lorig = []
    lexp = []
    lrate = []
    lcondtype = []
    lcompval = []
    lgentype = []
    lansincl = []
    for comb in lcomb:
        # print(comb)
        condtype = comb["condtype"]
        lcompare = comb["lcompare"]
        gentype = comb["gentype"]
        ov, ev, ir, ar = avgllist_gentype_ansinc(
            dfs, condtype, lcompare, gentype, **dlist
        )
        lorig.append(ov)
        lexp.append(ev)
        lrate.append(ir)
        lansincl.append(ar)
        lcondtype.append(condtype)
        lcompval.append(lcompare[0].replace("orig_", ""))
        lgentype.append(gentype)
        # print(lcondtype)
        # print(lcompval)

    llabel = lgentype
    ylabel = lcondtype[0]
    commonval = lcompval[0]
    # ax.set_title(f"{lcondtype[0]}")

    # print(llabel)
    y1minlim = ax.get_ylim()[0]
    ax.grid(True)

    ax.set(ylim=(y1minlim, y1maxlim))
    ax2 = ax.twinx()
    y2minlim = min(0, min(lrate))
    ax2.set(ylim=(y2minlim, y2maxlim))
    data = []
    for i, (orig, exp, rate, aninr, condtype, compval, label) in enumerate(
        zip(lorig, lexp, lrate, lansincl, lcondtype, lcompval, llabel)
    ):
        data.append(
            {
                "val": orig,
                "class": "Original",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
        data.append(
            {
                "val": exp,
                "class": "Explicitation",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
        data.append(
            {
                "val": rate * (y1maxlim - y1minlim) / (y2maxlim - y2minlim),
                "class": "Increase Rate",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
        data.append(
            {
                "val": aninr * (y1maxlim - y1minlim) / (y2maxlim - y2minlim),
                "class": "Answer Inclusion Rate",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
    print(data)
    d = pd.DataFrame(data)
    ax = sns.barplot(
        x="label",
        y="val",
        data=d,  # [(d["class"] == "orig") | (d["class"] == "exp")],
        capsize=0.3,
        # dodge=False,
        ax=ax,
        zorder=1,
        # color="b",
        hue="class",
        palette="pastel",
    )
    ax.legend(title="", loc=loc)  # to remove legend title

    ax.set_title(f"{title_prefix}{NAMECONV[ylabel]}")
    ax.set_xlabel("")  # xlabel
    ax.set_ylabel(NAMECONV[commonval])
    ax2.set_ylabel("Increase Rate(%)")
    if ylabel == "ewo" or turnofflegend:
        ax.legend([], [], frameon=False)  # turnoff legend
    for i, p in enumerate(ax.patches):
        if i > 5:
            ax.annotate(
                f"{p.get_height() * (y2maxlim - y2minlim) / (y1maxlim - y1minlim):.0f}%",
                (p.get_x(), p.get_height() + ax.get_ylim()[1] * 0.03),
            )
        else:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() - 0.05, p.get_height() + ax.get_ylim()[1] * 0.03),
            )


def plot_compare_bar_gentype(
    ax,
    lcomb,
    dfs,
    title_prefix=None,
    loc="lower right",
    turnofflegend=False,
    y1maxlim=0.75,
    y2maxlim=250,
    **dlist,
):
    lorig = []
    lexp = []
    lrate = []
    lcondtype = []
    lcompval = []
    lgentype = []
    for comb in lcomb:
        # print(comb)
        condtype = comb["condtype"]
        lcompare = comb["lcompare"]
        gentype = comb["gentype"]
        ov, ev, ir = avgllist_gentype(dfs, condtype, lcompare, gentype, **dlist)
        lorig.append(ov)
        lexp.append(ev)
        lrate.append(ir)
        lcondtype.append(condtype)
        lcompval.append(lcompare[0].replace("orig_", ""))
        lgentype.append(gentype)
        # print(lcondtype)
        # print(lcompval)

    llabel = lgentype
    ylabel = lcondtype[0]
    commonval = lcompval[0]
    # ax.set_title(f"{lcondtype[0]}")

    # print(llabel)
    y1minlim = ax.get_ylim()[0]
    ax.grid(True)

    ax.set(ylim=(y1minlim, y1maxlim))
    ax2 = ax.twinx()
    y2minlim = min(0, min(lrate))
    ax2.set(ylim=(y2minlim, y2maxlim))
    data = []
    for i, (orig, exp, rate, condtype, compval, label) in enumerate(
        zip(lorig, lexp, lrate, lcondtype, lcompval, llabel)
    ):
        data.append(
            {
                "val": orig,
                "class": "Original",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
        data.append(
            {
                "val": exp,
                "class": "Explicitation",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
        data.append(
            {
                "val": rate * (y1maxlim - y1minlim) / (y2maxlim - y2minlim),
                "class": "Increase Rate",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
    print(data)
    d = pd.DataFrame(data)
    ax = sns.barplot(
        x="label",
        y="val",
        data=d,  # [(d["class"] == "orig") | (d["class"] == "exp")],
        capsize=0.3,
        # dodge=False,
        ax=ax,
        zorder=1,
        # color="b",
        hue="class",
        palette="pastel",
    )
    ax.legend(title="")  # to remove legend title

    ax.set_title(f"{title_prefix}{NAMECONV[ylabel]}")
    ax.set_xlabel("")  # xlabel
    ax.set_ylabel(NAMECONV[commonval])
    ax2.set_ylabel("Increase Rate(%)")
    if ylabel == "ewo":
        ax.legend([], [], frameon=False)  # turnoff legend
    for i, p in enumerate(ax.patches):
        if i > 5:
            ax.annotate(
                f"{p.get_height() * (y2maxlim - y2minlim) / (y1maxlim - y1minlim):.0f}%",
                (p.get_x(), p.get_height() + ax.get_ylim()[1] * 0.03),
            )
        else:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() - 0.05, p.get_height() + ax.get_ylim()[1] * 0.03),
            )


def plot_compare_bar2(
    ax, lcomb, dfs, title_prefix=None, y1maxlim=0.75, y2maxlim=250, **dlist
):
    lorig = []
    lexp = []
    lrate = []
    lcondtype = []
    lcompval = []
    for comb in lcomb:
        # print(comb)
        condtype = comb["condtype"]
        lcompare = comb["lcompare"]
        ov, ev, ir = average_list(dfs, condtype, lcompare, **dlist)
        lorig.append(ov)
        lexp.append(ev)
        lrate.append(ir)
        lcondtype.append(condtype)
        lcompval.append(lcompare[0].replace("orig_", ""))
        # print(lcondtype)
        # print(lcompval)
    if len(list(set(lcondtype))) == 1:
        llabel = lcompval
        ylabel = f"{lcondtype[0]}"
        xlabel = "Metrics"
        ax.set_title(f"{lcondtype[0]}")

    elif len(list(set(lcompval))) == 1:
        llabel = lcondtype
        ylabel = f"{lcompval[0]}"
        xlabel = "Scope"
        ax.set_title(f"{lcompval[0]}")
    else:
        raise NotImplementedError
    # print(llabel)
    y1minlim = ax.get_ylim()[0]
    ax.grid(True)
    if ylabel == "ewo":
        y2maxlim = 50
    ax.set(ylim=(y1minlim, y1maxlim))
    ax2 = ax.twinx()
    y2minlim = min(0, min(lrate))
    ax2.set(ylim=(y2minlim, y2maxlim))
    data = []
    for i, (orig, exp, rate, condtype, compval, label) in enumerate(
        zip(lorig, lexp, lrate, lcondtype, lcompval, llabel)
    ):
        data.append(
            {
                "val": orig,
                "class": "Original",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
        data.append(
            {
                "val": exp,
                "class": "Explicitation",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
        data.append(
            {
                "val": rate * (y1maxlim - y1minlim) / (y2maxlim - y2minlim),
                "class": "Increase Rate",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": NAMECONV[label],
            }
        )
    d = pd.DataFrame(data)
    ax = sns.barplot(
        x="label",
        y="val",
        data=d,  # [(d["class"] == "orig") | (d["class"] == "exp")],
        capsize=0.3,
        # dodge=False,
        ax=ax,
        zorder=1,
        # color="b",
        hue="class",
        palette="pastel",
    )
    ax.legend(title="")  # to remove legend title

    ax.set_title(f"{title_prefix}{NAMECONV[ylabel]}")
    ax.set_xlabel("")  # xlabel
    ax.set_ylabel(NAMECONV[ylabel])
    ax2.set_ylabel("Increase Rate(%)")
    if ylabel == "ewo":
        ax.legend([], [], frameon=False)  # turnoff legend
    for i, p in enumerate(ax.patches):
        if i > 5:
            ax.annotate(
                f"{p.get_height() * (y2maxlim - y2minlim) / (y1maxlim - y1minlim):.0f}%",
                (p.get_x(), p.get_height() + ax.get_ylim()[1] * 0.03),
            )
        else:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x() - 0.05, p.get_height() + ax.get_ylim()[1] * 0.03),
            )


def plot_compare_bar(ax, lcomb, dfs, **dlist):
    lorig = []
    lexp = []
    lrate = []
    lcondtype = []
    lcompval = []
    for comb in lcomb:
        # print(comb)
        condtype = comb["condtype"]
        lcompare = comb["lcompare"]
        ov, ev, ir = average_list(dfs, condtype, lcompare, **dlist)
        lorig.append(ov)
        lexp.append(ev)
        lrate.append(ir)
        lcondtype.append(condtype)
        lcompval.append(lcompare[0].replace("orig_", ""))
        # print(lcondtype)
        # print(lcompval)
    if len(list(set(lcondtype))) == 1:
        llabel = lcompval
        ax.set_title(f"{lcondtype[0]}")

    elif len(list(set(lcompval))) == 1:
        llabel = lcondtype
        ax.set_title(f"{lcompval[0]}")
    else:
        raise NotImplementedError
    # print(llabel)
    data = []
    for i, (orig, exp, rate, condtype, compval, label) in enumerate(
        zip(lorig, lexp, lrate, lcondtype, lcompval, llabel)
    ):
        data.append(
            {
                "val": orig,
                "class": "orig",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": label,
            }
        )
        data.append(
            {
                "val": exp,
                "class": "exp",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": label,
            }
        )
        data.append(
            {
                "val": rate / 100,
                "class": "inc_rate",
                "rate": rate,
                "condtype": condtype,
                "compval": compval,
                "label": label,
            }
        )
    d = pd.DataFrame(data)
    ax = sns.barplot(
        x="label",
        y="val",
        data=d,
        capsize=0.3,
        # dodge=False,
        ax=ax,
        zorder=1,
        # color="b",
        hue="class",
        palette="pastel",
    )
    for i, p in enumerate(ax.patches):
        if i > 5:
            ax.annotate(
                f"{p.get_height()*100:.0f}%",
                (p.get_x(), p.get_height() + ax.get_ylim()[1] * 0.03),
            )
        else:
            ax.annotate(
                f"{p.get_height():.2f}",
                (p.get_x(), p.get_height() + ax.get_ylim()[1] * 0.03),
            )


def list_in_range(
    lcompare, dfs, maxv=1000000000, minv=0, qid_filter=None, key="ent_id"
):
    if qid_filter:
        qa_list = list(
            dfs[
                (dfs[lcompare[1]] - dfs[lcompare[0]] > minv)
                & (dfs[lcompare[1]] - dfs[lcompare[0]] <= maxv)
                & (dfs["orig_qid"].isin(qid_filter))
            ][key]
        )
    else:
        qa_list = list(
            dfs[
                (dfs[lcompare[1]] - dfs[lcompare[0]] > minv)
                & (dfs[lcompare[1]] - dfs[lcompare[0]] <= maxv)
            ][key]
        )
    # print(f"{qa_list=} {minv=}")
    return qa_list


def binning(lcompare, dfs, l_id_ours, qid_filter=None, nbin=5):
    maxv = max(dfs["exp_mrr_eci"] - dfs["orig_mrr_eci"])
    minv = max(0, min(dfs["exp_mrr_eci"] - dfs["orig_mrr_eci"]))
    lthr_range = [
        ((maxv - minv) / nbin * i, (maxv - minv) / nbin * (i + 1)) for i in range(nbin)
    ]
    lthr = [(minv, maxv)] + lthr_range
    l_n_eid_qa = []
    l_n_id_ours_in_qa = []
    for pmin, pmax in lthr:
        l_eid_qa = list_in_range(
            lcompare, dfs, maxv=pmax, minv=pmin, qid_filter=qid_filter, key="ent_id"
        )
        l_n_eid_qa.append(len(l_eid_qa))
        l_id_ours_in_qa = [eid for eid in l_eid_qa if eid in l_id_ours]
        l_n_id_ours_in_qa.append(len(l_id_ours_in_qa))
    return lthr, l_n_eid_qa, l_n_id_ours_in_qa


def plot_prfchanges(
    ax, lcompare, dfs, l_ent_id_related, qid_filter=None, loc="lower right", ylimax=0.5
):
    lthr, lp, lr, lf = prfchanges(
        lcompare, dfs, l_ent_id_related, qid_filter=qid_filter
    )

    ax.set(ylim=(0.0, ylimax))
    ax = sns.lineplot(y=lp, x=lthr, ax=ax, label="precision")
    ax = sns.lineplot(y=lr, x=lthr, ax=ax, label="recall")
    ax = sns.lineplot(y=lf, x=lthr, ax=ax, label="f1")
    ax.grid(True)
    ax.set_title(f"{lcompare[0]} vs {lcompare[1]}")


def prf(lcompare, dfs, our_list, thr=0, qid_filter=None, key="ent_id"):
    if qid_filter:
        qa_list = list(
            dfs[
                (dfs[lcompare[1]] - dfs[lcompare[0]] > thr)
                & (dfs["orig_qid"].isin(qid_filter))
            ][key]
        )
    else:
        qa_list = list(dfs[dfs[lcompare[1]] - dfs[lcompare[0]] > thr][key])
    dict_prf = precision2(qa_list, our_list, False)
    return dict_prf["precision"], dict_prf["recall"], dict_prf["f1"]


def prfchanges(lcompare, dfs, our_list, qid_filter=None, nbin=20):
    maxv = max(dfs["exp_mrr_eci"] - dfs["orig_mrr_eci"])
    minv = max(0, min(dfs["exp_mrr_eci"] - dfs["orig_mrr_eci"]))
    lthr = [(maxv - minv) / nbin * 0.5 * i for i in range(nbin)]
    lp, lr, lf = [], [], []
    for thr in lthr:
        p, r, f = prf(lcompare, dfs, our_list, thr, qid_filter=qid_filter, key="ent_id")
        lp.append(p)
        lr.append(r)
        lf.append(f)
    return lthr, lp, lr, lf
