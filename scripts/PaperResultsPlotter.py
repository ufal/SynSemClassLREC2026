#!/usr/bin/env python3

# Copyright 2025 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""PaperResultsPlotter class for plotting paper-ready tables and figures."""


import argparse
from collections import defaultdict
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytrec_eval
from statsmodels.stats.inter_rater import fleiss_kappa
import statsmodels.api as sm

from IREvaluator import IREvaluatorWithSuggestions, IREvaluatorWithoutSuggestions, METRIC_NAMES, METRIC_NAMES_SHORT
from IAAEvaluator import IAAEvaluator


class PaperResultsPlotter():
    """PaperResultsPlotter class for plotting paper-ready tables and figures."""


    def __init__(self, args):
        self._args = args
        self._K = 5
        self._task1_ir_evaluator = IREvaluatorWithoutSuggestions(annotation_filenames=args.task1_annotations,
                                                                 excluded_filename=args.task1_excluded,
                                                                 gold_filename=args.task1_gold,
                                                                 lemma_suggestions_filename=args.task1_lemma_suggestions,
                                                                 num_frames_filename=args.num_frames,
                                                                 task="Task 1",
                                                                 verbose=args.verbose)
        self._task2_ir_evaluator = IREvaluatorWithSuggestions(annotation_filenames=args.task2_annotations,
                                                              excluded_filename=args.task2_excluded,
                                                              frame_suggestions_filename=args.task2_frame_suggestions,
                                                              gold_filename=args.task2_gold,
                                                              lemma_suggestions_filename=args.task2_lemma_suggestions,
                                                              num_frames_filename=args.num_frames,
                                                              max_annotated_rows=args.task2_max_annotated_rows,
                                                              task="Task 2",
                                                              verbose=args.verbose)
        self._task3_ir_evaluator = IREvaluatorWithSuggestions(annotation_filenames=args.task3_annotations,
                                                              excluded_filename=args.task3_excluded,
                                                              gold_filename=args.task3_gold,
                                                              lemma_suggestions_filename=args.task3_lemma_suggestions,
                                                              num_frames_filename=args.num_frames,
                                                              task="Task 3",
                                                              verbose=args.verbose)

        iaa_evaluator = IAAEvaluator(args)
        self._iaa_results = iaa_evaluator.evaluate()


    def plot(self):
        self._print_suggestions_quality_table()
        self._print_task_stats()
        self._plot_metric_by_frames_per_lemma(metric="set_recall", K=self._K)
        self._plot_metric_by_K(metric="set_recall")


    def _print_suggestions_quality_table(self):
        output_filename="suggestions_quality.tex"
        print(f"Printing LaTeX tables to {output_filename}")
        with open(output_filename, "w", encoding="utf-8") as fw:

            print("\\begin{table*}", file=fw)
            print("\t\\begin{centering}", file=fw)
            print("\t\\begin{tabular}{lcccc}", file=fw)
            print("\t\t\\toprule", file=fw)
            print("\t\t             & Task 1 & Task 2 & Task 2 & Task 3 \\\\", file=fw)
            print("\t\t Ambiguity   & High & High & Low & Low \\\\", file=fw)
            print("\t\t Suggestions & Without & With & With & With \\\\", file=fw)
            print("\t\t Type        & L & F & L+F & L \\\\", file=fw)
            print("\t\t\midrule", file=fw)

            for gold_side in ["annotations", "gold"]:
                print("\t\t\multicolumn{5}{c}{Suggestions vs. " + f"{gold_side}" + "{}".format(" (up to 5 selected)" if gold_side == "annotations" else " (1 selected)") + "} \\\\", file=fw)

                task1_lemma_results = self._task1_ir_evaluator.evaluate(gold_side=gold_side,
                                                                        K=self._K)
                task2_high_amb_results = self._task2_ir_evaluator.evaluate(frame_range=list(range(3, 200)),
                                                                           gold_side=gold_side,
                                                                           K=self._K,
                                                                           suggestion_type=None)

                task2_low_amb_results = self._task2_ir_evaluator.evaluate(frame_range=[1,2],
                                                                          gold_side=gold_side,
                                                                          K=self._K,
                                                                          suggestion_type=None)
                task3_lemma_results = self._task3_ir_evaluator.evaluate(gold_side=gold_side,
                                                                        K=self._K,
                                                                        suggestion_type=None)


                for metric in ["set_recall", "map"]:
                    print("\t\t{} & {:.2f}$^\\dag$ & {:.2f} & {:.2f} & {:.2f} \\\\".format(METRIC_NAMES_SHORT[metric],
                                                                                           100*task1_lemma_results[metric],
                                                                                           100*task2_high_amb_results[metric],
                                                                                           100*task2_low_amb_results[metric],
                                                                                           100*task3_lemma_results[metric]), file=fw)
                if gold_side == "annotations":
                    print("\t\t\midrule", file=fw)

            print("\t\t\\bottomrule", file=fw)
            print("\t\\end{tabular}", file=fw)
            print("\t\\caption{Recall and Mean Average Precision (MAP) at rank 5 of automatic suggestions quality for low- and high- semantic ambiguity conditions, measured against four annotators (upper) and gold data by expert annotator (lower). ``F'' stands for frame-based suggestions from PDT-C 2.0, ``L'' for lemma-based suggestions from SYN v4. $^\dag$ marks annotation without suggestions evaluated with suggestions generated ex-post.}", file=fw)
            print("\t\\label{tab:suggestions-quality}", file=fw)
            print("\t\\end{centering}", file=fw)
            print("\\end{table*}", file=fw)

            print("", file=fw, flush=True)


    def _print_task_stats(self):
        output_filename="task_stats.tex"
        print(f"Printing LaTeX tables to {output_filename}")
        with open(output_filename, "w", encoding="utf-8") as fw:

            print("\\begin{table}", file=fw)
            print("\t\\begin{centering}", file=fw)
            print("\t\\begin{tabular}{lcccc}", file=fw)
            print("\t\t\\toprule", file=fw)
            print("\t\t           & Task 1 & Task 2 & Task 2 & Task 3 \\\\", file=fw)
            print("\t\t Ambiguity & High   & High & Low & Low \\\\", file=fw)
            print("\t\t\midrule", file=fw)
            print("\t\tAnnotated  & 463    & 614  & 380 & 354  \\\\", file=fw)
            print("\t\tShared     & {}     & {}   & {}  & {}   \\\\".format(self._iaa_results["Task 1"]["N"],
                                                                            self._iaa_results["Task 2 High-Amb"]["N"],
                                                                            self._iaa_results["Task 2 Low-Amb"]["N"],
                                                                            self._iaa_results["Task 3"]["N"]), file=fw)
            print("\t\tAnnotators & 6      & 4    & 4   & 4    \\\\", file=fw)
            print("\t\t\\bottomrule", file=fw)
            print("\t\\end{tabular}", file=fw)
            print("\t\\caption{Number of frames annotated by at least one annotator, number of shared frames (valid for the study), and number of annotators in the shared annotations, across all tasks.}", file=fw)
            print("\t\\label{tab:task_stats}", file=fw)
            print("\t\\end{centering}", file=fw)
            print("\\end{table}", file=fw)


    def _plot_metric_by_frames_per_lemma(self, metric="set_recall", K=5):
        """LaTeX table and figure for metric decomposed by frames per lemma."""

        # Collect the scores
        x_values = []
        x_labels = []
        y1_lemma_scores, y2_lemma_scores, y2_frame_scores = [], [], []
        y1_lemma_counts, y2_lemma_counts, y2_frame_counts = [], [], []
        for i in range(1, 11):
            frame_range=[i] if i < 10 else [j for j in range(10,200)]

            # Suggestions vs. annotations
            task1_lemma_results = self._task1_ir_evaluator.evaluate_suggestions_vs_annotations(frame_range=frame_range,
                                                                                               K=K,
                                                                                               suggestion_type="lemma")
            task2_lemma_results = self._task2_ir_evaluator.evaluate_suggestions_vs_annotations(frame_range=frame_range,
                                                                                               K=K,
                                                                                               suggestion_type="lemma")
            task2_frame_results = self._task2_ir_evaluator.evaluate_suggestions_vs_annotations(frame_range=frame_range,
                                                                                               K=K,
                                                                                               suggestion_type="frame")

            x_values.append(i)
            x_labels.append(str(i) if i < 10 else "10+")

            for results, scores, counts in zip ([task1_lemma_results, task2_lemma_results, task2_frame_results],
                                                [y1_lemma_scores, y2_lemma_scores, y2_frame_scores],
                                                [y1_lemma_counts, y2_lemma_counts, y2_frame_counts]):

                if results and results[metric] and results["num_annotated_frames"] >= self._args.plot_frames_per_lemma_min_samples:
                    scores.append(100*results[metric])
                    counts.append(results["num_annotated_frames"])
                else:
                    scores.append(np.nan)
                    counts.append(0)

        # Plot figure
        legend = ["Task 1 High (L)", "Task 2 Low (L)", "Task 2 Low+High (F)"]
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

        ax1.plot([x for (x, y) in zip(x_values, y1_lemma_scores) if not np.isnan(y)], [y for y in y1_lemma_scores if not np.isnan(y)], marker="o", label=legend[0])
        ax1.plot([x for (x, y) in zip(x_values, y2_lemma_scores) if not np.isnan(y)], [y for y in y2_lemma_scores if not np.isnan(y)], marker="s", label=legend[1])
        ax1.plot([x for (x, y) in zip(x_values, y2_frame_scores) if not np.isnan(y)], [y for y in y2_frame_scores if not np.isnan(y)], marker="^", label=legend[2])
        ax1.set_ylabel(METRIC_NAMES[metric])
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.6)

        max_height = max(np.array(y1_lemma_counts) + np.array(y2_lemma_counts) + np.array(y2_frame_counts))
        ax2.set_ylim(0, max_height * 1.05)
        bar_width=0.5
        ax2.bar(x_values, y1_lemma_counts, label=legend[0], width=bar_width)
        ax2.bar(x_values, y2_lemma_counts, label=legend[1], bottom=np.array(y1_lemma_counts), width=bar_width)
        ax2.bar(x_values, y2_frame_counts, label=legend[2], bottom=np.array(y1_lemma_counts)+np.array(y2_lemma_counts), width=bar_width)
        ax2.set_ylabel("Frames annotated")
        ax2.set_xticks(x_values)
        ax2.set_xticklabels(x_labels)
        ax2.set_xlabel("Number of frames per lemma")
        ax2.legend()

        plt.suptitle("{} and Data Distribution by Number of Frames per Lemma".format(METRIC_NAMES[metric]))
        plt.tight_layout()
        plt.savefig("{}-frames-per-lemma.pdf".format(metric))
        plt.close()


    def _plot_metric_by_K(self, metric="set_recall"):

        # Collect the scores
        x_values = []
        y2_low_amb_scores, y2_high_amb_scores, y3_low_amb_scores = [], [], []
        for i in range(1, 11):
            task2_low_amb_results = self._task2_ir_evaluator.evaluate_suggestions_vs_annotations(frame_range=[1,2],
                                                                                                 K=i,
                                                                                                 suggestion_type=None)
            task2_high_amb_results = self._task2_ir_evaluator.evaluate_suggestions_vs_annotations(frame_range=list(range(3,200)),
                                                                                                  K=i,
                                                                                                  suggestion_type=None)
            task3_low_amb_results = self._task3_ir_evaluator.evaluate_suggestions_vs_annotations(K=i,
                                                                                                 suggestion_type=None)

            x_values.append(i)
            y2_low_amb_scores.append(100*task2_low_amb_results[metric])
            y2_high_amb_scores.append(100*task2_high_amb_results[metric])
            y3_low_amb_scores.append(100*task3_low_amb_results[metric])

        plt.figure()
        plt.plot(x_values, y2_low_amb_scores, marker="s", label="Task 2 Low (L+F)")
        plt.plot(x_values, y2_high_amb_scores, marker="o", label="Task 2 High (F)")
        plt.plot(x_values, y3_low_amb_scores, marker="^", label="Task 3 Low (L)")
        plt.axvline(x=5, color="gray", linestyle="--", label="Presented suggestions")
        plt.xlabel("Number of retrieved suggestions per frame (K)")
        plt.xticks(list(range(1,11)))
        plt.ylabel("Recall")
        plt.title("Recall at K")
        plt.legend()
        plt.tight_layout()
        plt.savefig("{}-at-K.pdf".format(metric))
        plt.close()
