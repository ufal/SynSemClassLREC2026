#!/usr/bin/env python3

# Copyright 2025 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""Evaluation of automatic annotation suggestions for the SynSemClass ontology.

The inputs into the annotation pipeline were valency frames (i.e., verb senses)
and the goal was to sort the input frames into the SynSemClass ontology
semantic classes.

Additionally, the automatic suggestions were sourced from two origins: either
based on lemma in a large unannotated corpus (Czech SYN v4) or based directly
on the valency frame in a smaller corpus with manually annoated valency frames
(Czech PDT-C 2.0).

The script computes various metrics to assess the efficiency of the automatic
suggestions in all scenarios (Task 1/Task 2/Task 3 x lemmas/frames).

In order to run this script, you'll need some extra Python packages (i.e.,
pandas) installed into Python virtual environment:

python3 -m venv venv
venv/bin/pip install -r requirements.txt

Example Usage:

# Print help:
venv/bin/python ./evaluate.py --help

# Generate publication-ready tables and figures for the paper:
venv/bin/python ./evaluate.py --paper_results

# Analyze IAA and accuracy to gold data:
venv/bin/python ./evaluate.py --iaa

# Task 1, all verbs, lemma suggestions
venv/bin/python ./evaluate.py --task=task1 --suggestion_type=lemma

# Task 2, all verbs, lemma suggestions
venv/bin/python ./evaluate.py --task=task2 --suggestion_type=lemma

# Task 2, all verbs, frame suggestions
venv/bin/python ./evaluate.py --task=task2 --suggestion_type=frame

# Task3, all verbs, lemma suggestions
venv/bin/python ./evaluate.py --task=task3 --suggestion_type=lemma

# All tasks, all verbs, lemma suggestions
venv/bin/python ./evaluate.py --task=all --suggestion_type=lemma

# All tasks, all verbs, both lemma and frame suggestions
venv/bin/python ./evaluate.py --task=all --suggestion_type=all
"""


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

from IREvaluator import IREvaluatorWithoutSuggestions, IREvaluatorWithSuggestions, METRIC_NAMES, METRIC_NAMES_SHORT
from PaperResultsPlotter import PaperResultsPlotter
from IAAEvaluator import IAAEvaluator


def parse_range(s):
    nums = []
    for part in s.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            nums.extend(range(start, end+1))
        else:
            nums.append(int(part))
    return nums


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General arguments
    parser.add_argument("--frame_range", type=parse_range, default=None, help="Comma-separated list of ranges of number of framer per lemma (e.g., 1,2,4-6).")
    parser.add_argument("--gold_side", type=str, choices=["annotations", "gold"], default="annotations")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--num_frames", type=str, default="../data/pocet_ramcu_pro_sloveso", help="File with number of frames per lemma.")
    parser.add_argument("--paper_results", action="store_true", default=False, help="Generate publication-ready tables and figures for the paper.")
    parser.add_argument("--plot_frames_per_lemma_min_samples", type=int, default=7, help="Minimum samples for plotting the frames per lemma figure.")
    parser.add_argument("--pred_side", type=str, choices=["suggestions", "annotations"], default="suggestions")
    parser.add_argument("--print_supported_measures", action="store_true", default=False)
    parser.add_argument("--iaa", action="store_true", default=False, help="Analyze IAA.")
    parser.add_argument("--regression_analysis_min_samples", type=int, default=5, help="Number of minimum samples per analysis bin.")
    parser.add_argument("--print_histogram", action="store_true", default=False, help="Prints histogram of frames per lemma for the given task and suggestion type.")
    parser.add_argument("--suggestion_type", type=str, choices=["lemma", "frame"], default="lemma", help="For Task 2, evaluate either lemma or frame suggestions.")
    parser.add_argument("--task", type=str, choices=["task1", "task2", "task3", "all"], default="task2", help="Task 1, Task 2, Task 3 or all.")
    parser.add_argument("--verbose", type=int, choices=[0, 1, 2, 3], default=1, help="Set verbosity level: 0=errors only, 1=normal, 2=verbose, 3=debug")

    # Task 1 arguments
    parser.add_argument("--task1_annotations", type=str, default="../data/task1/A1_First_task.xlsx,../data/task1/A2_First_task.xlsx,../data/task1/A3_First_task.xlsx,../data/task1/A4_First_task.xlsx,../data/task1/A5_First_task.xlsx,../data/task1/A6_First_task.xlsx", help="Task 1 annotations")
    parser.add_argument("--task1_excluded", type=str, default="../data/task1/vyhozene_ramce.xlsx", help="Frames excluded from Task 1 in a XLSX file.")
    parser.add_argument("--task1_gold", type=str, default="../data/task1/task1_gold_upraveny_pro_lrec.csv", help="Task 1 gold data in TSV (tab-separated values) format.")
    parser.add_argument("--task1_lemma_suggestions", type=str, default="../data/task1/task1_synv4_lemma_suggestions_n=10.tsv", help="Task 1 automatic suggestions TSV file.")

    # Task 2 arguments
    parser.add_argument("--task2_max_annotated_rows", type=int, default=1025)
    parser.add_argument("--task2_annotations", type=str, default="../data/task2/A1_SSC.5.5_frame_suggestions_07_25.xlsx,../data/task2/A2_SSC.5.5_frame_suggestions_07_25.xlsx,../data/task2/A3_SSC.5.5_frame_suggestions_07_25.xlsx,../data/task2/A5_SSC.5.5_frame_suggestions_07_25.xlsx", help="Task 2 annotation XLSX file(s).")
    parser.add_argument("--task2_frame_suggestions", type=str, default="../data/task2/task2_pdtc20_frame_suggestions_n=10.tsv", help="Task 2 automatic frame suggestions TSV file.")
    parser.add_argument("--task2_gold", type=str, default="../data/task2/task2_gold_upraveny_pro_lrec.csv", help="Task 2 gold data in TSV (tab-separated values) format.")
    parser.add_argument("--task2_lemma_suggestions", type=str, default="../data/task2/task2_synv4_lemma_suggestions_n=10.tsv", help="Task 2 automatic lemma suggestions TSV file.")
    parser.add_argument("--task2_excluded", type=str, default=None, help="Frames excluded from Task 2 in a XLSX file.")

    # Task 3 arguments
    parser.add_argument("--task3_annotations", type=str, default="../data/task3/A1_SSC.5.5_only_lemma_suggestions_07_25.xlsx,../data/task3/A2_SSC.5.5_only_lemma_suggestions_07_25.xlsx,../data/task3/A3_SSC.5.5_only_lemma_suggestions_07_25.xlsx,../data/task3/A5_SSC.5.5_only_lemma_suggestions_07_25.xlsx", help="Task 3 annotation XLSX file(s).")
    parser.add_argument("--task3_excluded", type=str, default="../data/task3/vyhozene_ramce.xlsx", help="Frames excluded from Task 3 in a XLSX file.")
    parser.add_argument("--task3_gold", type=str, default="../data/task3/task3_gold_upraveny_pro_lrec.csv", help="Task 3 gold data in TSV (tab-separated values) format.")
    parser.add_argument("--task3_lemma_suggestions", type=str, default="../data/task3/task3_synv4_lemma_suggestions_n=10.tsv", help="Task 3 automatic lemma suggestions TSV file.")

    args = parser.parse_args()

    # Sanity check.
    if args.suggestion_type == "frame" and args.task in ["task1", "task3"]:
        raise ValueError(f"There were no frame suggestions in --task={args.task}, cannot use --suggestion_type=frame.")

    # Just print the pytrec eval supported measures and exit.
    if args.print_supported_measures:
        print(pytrec_eval.supported_measures)
        sys.exit()

    # Generate publication-ready tables and figures for the paper and exit.
    if args.paper_results:
        paper_results_plotter = PaperResultsPlotter(args)
        paper_results_plotter.plot()
        sys.exit()

    # Analyze IAA and Accuracy and exit.
    if args.iaa:
        iaa_evaluator = IAAEvaluator(args)
        iaa_evaluator.evaluate()
        sys.exit()

    # Compute IR metrics for Task 1.
    if args.task in ["task1", "all"]:
        ir_evaluator = IREvaluatorWithoutSuggestions(annotation_filenames=args.task1_annotations,
                                                     excluded_filename=args.task1_excluded,
                                                     gold_filename=args.task1_gold,
                                                     lemma_suggestions_filename=args.task1_lemma_suggestions,
                                                     num_frames_filename=args.num_frames,
                                                     task="Task 1",
                                                     verbose=args.verbose)
        results = ir_evaluator.evaluate(frame_range=args.frame_range,
                                        gold_side=args.gold_side,
                                        K=args.K,
                                        pred_side=args.pred_side,
                                        print_histogram=args.print_histogram)

        print("----------")
        ir_evaluator.print_metrics(results, K=args.K)
        print("----------")

    # Compute IR metrics for Task 2.
    if args.task in ["task2", "all"]:
        suggestion_types = ["lemma", "frame"] if args.suggestion_type == "all" else [args.suggestion_type]
        for suggestion_type in suggestion_types:
            ir_evaluator = IREvaluatorWithSuggestions(annotation_filenames=args.task2_annotations,
                                                      excluded_filename=args.task2_excluded,
                                                      frame_suggestions_filename=args.task2_frame_suggestions,
                                                      gold_filename=args.task2_gold,
                                                      lemma_suggestions_filename=args.task2_lemma_suggestions,
                                                      max_annotated_rows=args.task2_max_annotated_rows,
                                                      num_frames_filename=args.num_frames,
                                                      task="Task 2",
                                                      verbose=args.verbose)
            results = ir_evaluator.evaluate(suggestion_type=suggestion_type,
                                            frame_range=args.frame_range,
                                            gold_side=args.gold_side,
                                            K=args.K,
                                            pred_side=args.pred_side,
                                            print_histogram=args.print_histogram)

            print("----------")
            print("Macro Average Results:")
            ir_evaluator.print_metrics(results, K=args.K)
            print("----------")

    # Compute IR metrics for Task 3.
    if args.task in ["task3", "all"]:
        ir_evaluator = IREvaluatorWithSuggestions(annotation_filenames=args.task3_annotations,
                                                  excluded_filename=args.task3_excluded,
                                                  gold_filename=args.task3_gold,
                                                  lemma_suggestions_filename=args.task3_lemma_suggestions,
                                                  num_frames_filename=args.num_frames,
                                                  task="Task 3",
                                                  verbose=args.verbose)
        results = ir_evaluator.evaluate(frame_range=args.frame_range,
                                        gold_side=args.gold_side,
                                        K=args.K,
                                        pred_side=args.pred_side,
                                        suggestion_type=None)

        print("----------")
        print("Macro Average Results:")
        ir_evaluator.print_metrics(results, K=args.K)
        print("----------")
