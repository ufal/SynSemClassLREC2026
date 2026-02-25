#!/usr/bin/env python3

# Copyright 2025 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


"""IREvaluator class for IR metrics evaluation."""


import argparse
from collections import defaultdict
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import pytrec_eval
from statsmodels.stats.inter_rater import fleiss_kappa
import statsmodels.api as sm

from AnnotationsReader import AnnotationsReaderWithSuggestions, AnnotationsReaderWithoutSuggestions, GoldReader


# Nice metric names for printing.
METRIC_NAMES = {"map": "Mean average precision",
                "set_recall": "Recall",
                "num_rel": "Relevant classes",
                "num_ret": "Suggested classes",
                "num_rel_ret": "Correct suggestions",
                "avg_num_frames_per_lemma": "Frames per lemma",
                "num_annotated_frames": "Frames annotated",
                "num_pred_frames": "Frames predicted"}
METRIC_NAMES_SHORT = {"map": "MAP",
                      "set_recall": "Recall",
                      "num_rel": "Relevant classes",
                      "num_ret": "Suggested classes",
                      "num_rel_ret": "Correct suggestions",
                      "avg_num_frames_per_lemma": "Frames per lemma",
                      "num_annotated_frames": "Frames annotated",
                      "num_pred_frames": "Frames predicted"}


class IREvaluator():


    def __init__(self,
                 annotation_filenames=None,
                 excluded_filename=None,
                 frame_suggestions_filename=None,
                 gold_filename=None,
                 lemma_suggestions_filename=None,
                 max_annotated_rows=None,
                 num_frames_filename="../data/pocet_ramcu_pro_sloveso",
                 task="",
                 verbose=1):

        self._annotation_filenames = annotation_filenames.split(",") if annotation_filenames else []
        self._frame_suggestions_filename = frame_suggestions_filename
        self._gold_filename = gold_filename
        self._gold_reader = GoldReader(excluded_filename=excluded_filename, verbose=verbose)
        self._lemma_suggestions_filename = lemma_suggestions_filename
        self._max_annotated_rows = None
        self._num_frames_filename = num_frames_filename
        self._num_frames_df = pd.read_csv(self._num_frames_filename, sep="\t", header=None, names=["LemmaID", "Lemma", "NumFrames"])
        self._task = task
        self._verbose = verbose

        self._measures = ["map", "set_recall", "num_ret", "num_rel", "num_rel_ret"]


    def _evaluate_for_annotator(self, annotations_filename, gold_side="annotations", pred_side="suggestions", suggestion_type="lemma", frame_range=None, K=5, print_histogram=False):
        """Computes IR metrics for any task for one individual annotator."""

        if frame_range:
            selected_num_frames_df = self._num_frames_df[self._num_frames_df["NumFrames"].isin(frame_range)]
        else:
            selected_num_frames_df = None

        # Determine the correct reader based on the gold_side.
        if gold_side == "annotations":
            reader = self._annotations_reader
        elif gold_side == "gold":
            reader = self._gold_reader

        annotations_dict, lemma2frames = reader.get_annotations(annotations_filename,
                                                                suggestion_type=suggestion_type,
                                                                num_frames_df=selected_num_frames_df)

        # No annotations available for this setting
        if not annotations_dict:
            print("{}: WARNING: No annotations available for task {}{}{} from file: {}".format(type(self).__name__,
                                                                                              self._task,
                                                                                              " with suggestion_type: {}".format(suggestion_type if suggestion_type else ""),
                                                                                              " in frame_range: {}".format(frame_range if frame_range else ""),
                                                                                               annotations_filename))
            return None

        if suggestion_type == "frame":
            pred_dict = self._get_frame_suggestions(self._frame_suggestions_filename,
                                                   annotated_frames=annotations_dict.keys(),
                                                   K=K)
        elif suggestion_type == "lemma":
            pred_dict = self._get_lemma_suggestions(self._lemma_suggestions_filename,
                                                   lemma2frames=lemma2frames,
                                                   K=K)
        else:
            frame_pred_dict = self._get_frame_suggestions(self._frame_suggestions_filename,
                                                         annotated_frames=annotations_dict.keys(),
                                                         K=K)
            lemma_pred_dict = self._get_lemma_suggestions(self._lemma_suggestions_filename,
                                                         lemma2frames=lemma2frames,
                                                         K=K)
            pred_dict = self._merge_preds(frame_pred_dict, lemma_pred_dict)

        # Drop frames without the retrieved suggestions.
        if pred_side == "suggestions":
            before = len(annotations_dict.keys())
            annotations_dict = {k: v for k, v in annotations_dict.items() if k in pred_dict}
            after = len(annotations_dict.keys())
            if self._verbose >= 3:
                print("{}: Dropped {} (of {}) frame(s) without the retrieved suggestions, {} annotated frames remained.".format(type(self).__name__,
                                                                                                                                before-after,
                                                                                                                                before,
                                                                                                                                after))

        # Compute IR metrics.
        results = self._compute_ir_metrics(annotations_dict, pred_dict)

        if not results:
            return None

        # Add further information to results dict.
        results["num_annotated_frames"] = len(annotations_dict.keys())
        results["num_pred_frames"] = len(pred_dict.keys())

        num_frames_df = self._num_frames_df[self._num_frames_df["Lemma"].isin(lemma2frames.keys())]
        if print_histogram:
            print(num_frames_df["NumFrames"].value_counts().sort_index())
        results["avg_num_frames_per_lemma"] = sum(num_frames_df["NumFrames"]) / len(num_frames_df["NumFrames"])

        return results


    def evaluate_suggestions_vs_annotations(self, frame_range=None, K=5, print_histogram=False, suggestion_type="lemma"):
        """Convenience wrapper for evaluation of automatic suggestions to human annotations."""

        return self.evaluate(frame_range=frame_range,
                             gold_side="annotations",
                             K=K,
                             pred_side="suggestions",
                             print_histogram=print_histogram,
                             suggestion_type=suggestion_type)


    def evaluate_suggestions_vs_gold(self, frame_range=None, K=5, print_histogram=False, suggestion_type="lemma"):
        """Convenience wrapper for evaluation of automatic suggestions to human annotations."""

        return self.evaluate(frame_range=frame_range,
                             gold_side="gold",
                             K=K,
                             pred_side="suggestions",
                             print_histogram=print_histogram,
                             suggestion_type=suggestion_type)



    def evaluate(self, gold_side="annotations", pred_side="suggestions", suggestion_type="lemma", frame_range=None, K=5, print_histogram=False):
        """Computes IR metrics for any task over all annotators."""

        print("Evaluating {} {}from {} (K={}).".format(self._task, "for {}s ".format(suggestion_type) if suggestion_type else "", "SYN v4" if suggestion_type == "lemma" else "PDT-C 2.0", K))

        if frame_range:
            print("Frames per lemma restricted to {}.".format(", ".join(map(str, frame_range))))

        # Determine the file with gold data: annotations or gold.
        if gold_side == "annotations":
            annotation_filenames = self._annotation_filenames
        elif gold_side == "gold":
            annotation_filenames = [self._gold_filename]

        results = []
        for i, annotation_filename in enumerate(annotation_filenames):
            if self._verbose >= 2:
                print("----------")
                print("Annotator {}".format(i+1))

            results_per_annotator = self._evaluate_for_annotator(annotation_filename,
                                                                 gold_side=gold_side,
                                                                 pred_side=pred_side,
                                                                 suggestion_type=suggestion_type,
                                                                 frame_range=frame_range,
                                                                 K=K)

            if results_per_annotator:
                results.append(results_per_annotator)
                if self._verbose >= 2:
                   self.print_metrics(results[-1], K=K)

            if self._verbose >= 2:
                print("----------")

        macro_average_results = self.macro_average_metrics(results)
        return macro_average_results


    def print_metrics(self, results, K=5):
        """Prints IR metrics."""

        if not results:
            print("Results empty.")
            return

        for metric in sorted(results.keys()):
            if metric in ["map", "set_recall"]:
                print("{} @ {}:\t{:.2f}".format(metric, K, results[metric] * 100))
            elif metric in ["num_rel", "num_ret", "num_rel_ret"]:
                print("{} @ {}:\t{:d}".format(metric, K, int(results[metric])))
            elif metric in ["num_annotated_frames", "num_pred_frames", "avg_num_frames_per_lemma"]:
                print("{} @ {}:\t{:.2f}".format(metric, K, results[metric]))
            else:
                print("{} @ {}:\t{}".format(metric, K, results[metric]))


    def macro_average_metrics(self, dicts):
        """Computes macro average of IR metrics over dictionaries."""

        sums = defaultdict(float)
        counts = defaultdict(int)

        for d in dicts:
            for k, v in d.items():
                sums[k] += v
                counts[k] += 1

        return {k: sums[k] / counts[k] for k in sums}


    def _merge_preds(self, dict1, dict2):
        merged = {}

        # Get all outer keys from both dictionaries
        for key in set(dict1.keys()) | set(dict2.keys()):
            inner1 = dict1.get(key, {})
            inner2 = dict2.get(key, {})
            merged_inner = {}

            # Get all inner keys from both inner dictionaries
            for inner_key in set(inner1.keys()) | set(inner2.keys()):
                val1 = inner1.get(inner_key, float('-inf'))
                val2 = inner2.get(inner_key, float('-inf'))
                merged_inner[inner_key] = max(val1, val2)

            merged[key] = merged_inner

        return merged


    def _compute_ir_metrics(self, gold_dict, pred_dict):
        """Computes IR metrics using pytrec_eval."""

        for k in gold_dict:
            if k not in pred_dict:
                print(k)

        assert len(gold_dict.keys()) == len(pred_dict.keys()), "Number of frames with annotated gold class ({}) differs from number of frames with retrieved suggestions ({})".format(len(gold_dict.keys()), len(pred_dict.keys()))

        pytrec_eval_evaluator = pytrec_eval.RelevanceEvaluator(gold_dict, self._measures)
        results = pytrec_eval_evaluator.evaluate(pred_dict)

        # Compute aggregate scores
        aggregated_results = dict()
        for measure in self._measures:
            aggregated_results[measure] = pytrec_eval.compute_aggregated_measure(measure,
                                                                                 [query_measures[measure] for query_measures in results.values()])

        return aggregated_results


    def _get_lemma_suggestions(self, filename, lemma2frames, K=5):
        """Reads lemma suggestions."""

        if filename is None:
            if self._verbose >= 3:
                print("{}: No lemma suggestions file provided, reading 0 lemma suggestions.".format(type(self).__name__))
            return {}

        if self._verbose >= 3:
            print("Reading lemma suggestions from file {}.".format(filename))

        pred_dict = dict()
        df = pd.read_csv(filename, sep="\t", header=None)

        for _, row in df.iterrows():
            lemma = row.iloc[0]

            # Drop lemmas annotated with classes outside the current ontology.
            if lemma not in lemma2frames:
                continue

            for frame in lemma2frames[lemma]:
                inner_dict = {}

                count = 0
                for i in range(3, len(row), 2):  # loop over suggestion/value pairs
                    if count >= K:
                        break
                    suggestion = row.iloc[i]
                    value = row.iloc[i+1] if i+1 < len(row) else None
                    if pd.notna(suggestion):
                        inner_dict[suggestion] = value
                        count += 1

                pred_dict[frame] = inner_dict

        return pred_dict


    def _get_frame_suggestions(self, filename, annotated_frames, K=5):
        """Gets frame suggestions."""

        if filename is None:
            if self._verbose >= 3:
                print("{}: No frame suggestions file provided, reading 0 frame suggestions.".format(type(self).__name__))
            return {}

        if self._verbose >= 3:
            print("Reading frame suggestions from file {}".format(filename))

        pred_dict = dict()
        df = pd.read_csv(filename, sep="\t", header=None)

        for _, row in df.iterrows():
            frame = row.iloc[0]
            if frame not in annotated_frames:
                continue

            inner_dict = {}
            count = 0
            for i in range(3, len(row), 2):  # loop over suggestion/value pairs
                if count >= K:
                    break

                suggestion = row.iloc[i]
                value = row.iloc[i+1] if i+1 < len(row) else None
                if pd.notna(suggestion):
                    inner_dict[suggestion] = value
                    count += 1

            pred_dict[frame] = inner_dict

        return pred_dict


class IREvaluatorWithoutSuggestions(IREvaluator):


    def __init__(self,
                 annotation_filenames=None,
                 excluded_filename=None,
                 gold_filename=None,
                 lemma_suggestions_filename=None,
                 max_annotated_rows=None,
                 num_frames_filename="../data/pocet_ramcu_pro_sloveso",
                 task="",
                 verbose=1):

        self._annotations_reader = AnnotationsReaderWithoutSuggestions(excluded_filename=excluded_filename,
                                                                       max_annotated_rows=max_annotated_rows,
                                                                       verbose=verbose)

        super().__init__(annotation_filenames=annotation_filenames,
                         excluded_filename=excluded_filename,
                         gold_filename=gold_filename,
                         lemma_suggestions_filename=lemma_suggestions_filename,
                         max_annotated_rows=max_annotated_rows,
                         num_frames_filename=num_frames_filename,
                         task=task,
                         verbose=verbose)


class IREvaluatorWithSuggestions(IREvaluator):


    def __init__(self,
                 annotation_filenames=None,
                 excluded_filename=None,
                 frame_suggestions_filename=None,
                 gold_filename=None,
                 lemma_suggestions_filename=None,
                 max_annotated_rows=None,
                 num_frames_filename="../data/pocet_ramcu_pro_sloveso",
                 task="",
                 verbose=1):

        self._annotations_reader = AnnotationsReaderWithSuggestions(excluded_filename=excluded_filename,
                                                                    max_annotated_rows=max_annotated_rows,
                                                                    verbose=verbose)

        super().__init__(annotation_filenames=annotation_filenames,
                         excluded_filename=excluded_filename,
                         frame_suggestions_filename=frame_suggestions_filename,
                         gold_filename=gold_filename,
                         lemma_suggestions_filename=lemma_suggestions_filename,
                         max_annotated_rows=max_annotated_rows,
                         num_frames_filename=num_frames_filename,
                         task=task,
                         verbose=verbose)
