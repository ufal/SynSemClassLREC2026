#!/usr/bin/env python3

# Copyright 2025 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""IAAEvaluator class."""


import argparse
from collections import defaultdict
import os
import sys
import warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import pandas as pd
import pytrec_eval
from scipy.stats import ttest_ind
from statsmodels.stats.inter_rater import fleiss_kappa
import statsmodels.api as sm


from AnnotationsReader import AnnotationsReaderWithSuggestions, AnnotationsReaderWithoutSuggestions, GoldReader


class IAAEvaluator():
    """IAAEvaluator class."""


    def __init__(self, args):
        self._args = args
        self._num_frames_df = pd.read_csv(args.num_frames, sep="\t", header=None, names=["LemmaID", "Lemma", "NumFrames"])
        self._task1_reader = AnnotationsReaderWithoutSuggestions(excluded_filename=args.task1_excluded,
                                                                 max_annotated_rows=None,
                                                                 verbose=args.verbose)
        self._task2_reader = AnnotationsReaderWithSuggestions(excluded_filename=args.task2_excluded,
                                                              max_annotated_rows=args.task2_max_annotated_rows,
                                                              verbose=args.verbose)
        self._task3_reader = AnnotationsReaderWithSuggestions(excluded_filename=args.task3_excluded,
                                                              max_annotated_rows=None,
                                                              verbose=args.verbose)

        # Read Task 1 gold data
        task1_gold_reader = GoldReader(excluded_filename=args.task1_excluded,
                                       max_annotated_rows=None,
                                       verbose=args.verbose)
        self._task1_gold_df = self._read_gold(self._args.task1_gold, task1_gold_reader)


        # Read Task 2 gold_data
        task2_gold_reader = GoldReader(excluded_filename=args.task2_excluded,
                                       max_annotated_rows=None,
                                       verbose=args.verbose)
        self._task2_gold_df = self._read_gold(self._args.task2_gold, task2_gold_reader)

        # Read Task 3 gold data
        task3_gold_reader = GoldReader(excluded_filename=args.task3_excluded,
                                       max_annotated_rows=None,
                                       verbose=args.verbose)
        self._task3_gold_df = self._read_gold(self._args.task3_gold, task3_gold_reader)


    def evaluate(self):
        """Evaluate IAA and Accuracy."""

        annotations = []

        # Read Task 1 annotations
        annotations = self._read_annotations_without_suggestions(self._args.task1_annotations.split(","),
                                                                 annotations,
                                                                 task="Task 1")

        # Read Task 2 High-Amb annotations
        frame_range=list(range(3,200))
        selected_num_frames_df = self._num_frames_df[self._num_frames_df["NumFrames"].isin(frame_range)]
        annotations = self._read_annotations_with_suggestions(self._args.task2_annotations.split(","),
                                                              annotations,
                                                              self._task2_reader,
                                                              num_frames_df=selected_num_frames_df,
                                                              task="Task 2 High-Amb")

        # Read Task 2 Low-Amb annotations
        frame_range=[1,2]
        selected_num_frames_df = self._num_frames_df[self._num_frames_df["NumFrames"].isin(frame_range)]
        annotations = self._read_annotations_with_suggestions(self._args.task2_annotations.split(","),
                                                              annotations,
                                                              self._task2_reader,
                                                              num_frames_df=selected_num_frames_df,
                                                              task="Task 2 Low-Amb")

        # Read Task 3 annotations
        annotations = self._read_annotations_with_suggestions(self._args.task3_annotations.split(","),
                                                       annotations,
                                                       self._task3_reader,
                                                       task="Task 3")

        # Concatenate all annotations into one dataframe
        annotations_df = pd.concat(annotations)

        # Print task stats
        results = self._print_task_stats(annotations_df)

        # Compute accuracy for the regression analysis and t-test
        accuracies = []
        for task in ["Task 1", "Task 2 High-Amb", "Task 2 Low-Amb"]:
            for suggestions in ["With", "Without"]: # Suggestions predictor
                for i in range(1, 200): # NumFrames predictor

                    # Filter subset according to experimental conditions
                    df = annotations_df[(annotations_df["Task"] == task) & (annotations_df["Suggestions"] == suggestions) & (annotations_df["NumFrames"].isin([i]))]

                    if not len(df):
                        continue

                    # Get the corresponding gold data.
                    if task == "Task 1":
                        gold_df = self._task1_gold_df
                    elif task.startswith("Task 2"):
                        gold_df = self._task2_gold_df

                    # Compute accuracy.
                    accuracy, N = self._accuracy(gold_df, df, task)

                    # Drop underrepresented conditions with too few samples.
                    if N < self._args.regression_analysis_min_samples:
                        continue

                    if self._args.verbose >= 3:
                        print("Task: {}, Suggestions: {}, NumFrames: {}, N: {}, Accuracy: {:.2f}".format(task, suggestions, i, N, accuracy))

                    accuracies.append({"Task": task, "Suggestions": int(suggestions == "With"), "NumFrames": i, "Accuracy": accuracy, "N": N})

        acc_df = pd.DataFrame(accuracies)
        acc_df = acc_df.dropna(subset=["Accuracy"])

        # Compute t-test
        self._t_test(acc_df, metric="Accuracy", task1="Task 1", task2="Task 2 High-Amb")

        # Compute accuracy for the regression analysis and t-test
        iaa = []
        all_frames, all_dropped_frames = 0, 0
        for task in ["Task 1", "Task 2 High-Amb", "Task 2 Low-Amb", "Task 3"]:
            for suggestions in ["With", "Without"]: # Suggestions predictor
                for i in range(1, 200): # NumFrames predictor

                    # Filter subset according to experimental conditions
                    df = annotations_df[(annotations_df["Task"] == task) & (annotations_df["Suggestions"] == suggestions) & (annotations_df["NumFrames"].isin([i]))]
                    kappa, added_frames, dropped_frames = self._fleiss_kappa_from_df(df)
                    all_frames += added_frames
                    all_dropped_frames += dropped_frames

                    if not kappa:
                        continue

                    if self._args.verbose >= 3:
                        print("Task: {}, Suggestions: {}, NumFrames: {}, N: {}, Fleiss Kappa: {:.2f}".format(task, suggestions, i, added_frames, kappa))

                    iaa.append({"Task": task,
                                "Suggestions": int(suggestions == "With"),
                                "NumFrames": i,
                                "Ambiguity": int(i > 2), # 1 = High, 0 = Low
                                "IAA": kappa,
                                "N": added_frames})

        if self._args.verbose >= 3:
            print("Dropped {} ({:.2f}%) frames of {} due to underrepresented condition (--args.regression_analysis_min_samples={})".format(all_dropped_frames, 100*all_dropped_frames/all_frames, all_frames, self._args.regression_analysis_min_samples))

        iaa_df = pd.DataFrame(iaa)
        iaa_df = iaa_df.dropna(subset=["IAA"])

        # Compute t-test
        self._t_test(iaa_df, metric="IAA", task1="Task 1", task2="Task 2 High-Amb")

        # Run regression analysis
        self._run_regression_analysis(iaa_df)

        return results


    def _fleiss_kappa_from_df(self, df):
        if not len(df):
            return None, 0, 0

        # Pivot so each row = FrameID, each column = Annotator
        # Missing annotations will be filled with "MISSING"
        pivot = df.pivot(index="FrameID", columns="Annotator", values="Annotation")
        all_annotators = sorted(df["Annotator"].unique())
        pivot = pivot.reindex(columns=all_annotators, fill_value="MISSING")
        pivot = pivot.fillna("MISSING")

        # Determine all possible categories
        # Includes any annotations + 'MISSING' placeholder
        categories = sorted(pivot.stack().unique())
        cat_to_idx = {cat: i for i, cat in enumerate(categories)}

        # Build table of counts per category (Fleiss format)
        table = np.zeros((pivot.shape[0], len(categories)), dtype=int)

        # Drop underrepresented conditions with too few samples.
        if len(table) < self._args.regression_analysis_min_samples:
            return None, 0, len(table)

        for j, row in enumerate(pivot.values):
            for a in row:
                table[j, cat_to_idx[a]] += 1

        # Compute Fleiss' Kappa
        return fleiss_kappa(table), len(table), 0


    def _read_annotations_with_suggestions(self, filenames, annotations, reader, task="Task2", num_frames_df=None):

        for i, filename in enumerate(filenames):
            # Include both regular and weak annotations.
            task_dict, lemma2frame = reader.get_annotations(filename,
                                                            drop_deleted=True,
                                                            drop_frames_without_gold_class=False,
                                                            first_class_only=True,
                                                            num_frames_df=num_frames_df,
                                                            suggestion_type=None,
                                                            include_weak_annotations=True)
            df = pd.DataFrame([(k, list(v.keys())[0]) for k, v in task_dict.items()], columns=["FrameID", "Annotation"])

            lemma2frame_df = pd.DataFrame(list(lemma2frame.items()), columns=["Lemma", "FrameID"])
            for lemma, frames in lemma2frame.items():
                for frame in frames:
                    df.loc[df["FrameID"] == frame, "Lemma"] = lemma

            # Override weak annotations with regular annotations.
            task_dict, lemma2frame = reader.get_annotations(filename,
                                                            drop_deleted=True,
                                                            drop_frames_without_gold_class=False,
                                                            first_class_only=True,
                                                            num_frames_df=num_frames_df,
                                                            suggestion_type=None,
                                                            include_weak_annotations=False)

            for frame, frame_annotations in task_dict.items():
                assert len(frame_annotations) == 1
                for annotation in frame_annotations:
                    df.loc[df["FrameID"] == frame, "Annotation"] = annotation

            df["Annotator"] = str(i+1)
            df["Suggestions"] = "With"
            df["Task"] = task

            # Enrich the annotations with the number of frames per lemma
            df = df.merge(self._num_frames_df[["Lemma", "NumFrames"]], on="Lemma", how="left")

            # Unify annotation between tasks.
            df.loc[(df["Annotation"] == 0) | (df["Annotation"] == "0"), "Annotation"] = "OutKB"

            # Sanity checks
            if len(df[df["Annotation"] == "x"]):
                raise ValueError(f"Some of the annotated classes were not projected to the \"Annotation\" in {task}.")
            if len(df) == 0:
                raise ValueError(f"No data read for task {task} from file {filename}")

            annotations.append(df)

        return annotations


    def _read_annotations_without_suggestions(self, filenames, annotations, task="Task 1"):

        for i, filename in enumerate(filenames):
            task1_dict, lemma2frame = self._task1_reader.get_annotations(filename,
                                                                         drop_deleted=True,
                                                                         drop_frames_without_gold_class=False,
                                                                         first_class_only=True,
                                                                         num_frames_df=None,
                                                                         suggestion_type=None)

            df = pd.DataFrame([(k, list(v.keys())[0]) for k, v in task1_dict.items()], columns=["FrameID", "Annotation"])

            lemma2frame_df = pd.DataFrame(list(lemma2frame.items()), columns=["Lemma", "FrameID"])
            for lemma, frames in lemma2frame.items():
                for frame in frames:
                    df.loc[df["FrameID"] == frame, "Lemma"] = lemma

            df["Annotator"] = str(i+1)
            df["Suggestions"] = "Without"
            df["Task"] = task

            # Enrich the annotations with the number of frames per lemma
            df = df.merge(self._num_frames_df[["Lemma", "NumFrames"]], on="Lemma", how="left")

            # Unify annotation between tasks.
            df.loc[df["Annotation"] == "x", "Annotation"] = "OutKB"

            annotations.append(df)

        return annotations


    def _read_gold(self, filename, reader):

        task_dict, _ = reader.get_annotations(filename,
                                              drop_deleted=True,    # exclude "D", "P"
                                              drop_frames_without_gold_class=False, # keep "x"
                                              num_frames_df=None,
                                              suggestion_type=None)

        # Make DataFrame from dictionary.
        df = pd.DataFrame([(k, list(v.keys())[0]) for k, v in task_dict.items()], columns=["FrameID", "Annotation"])

        # Unify annotation between tasks.
        df.loc[df["Annotation"] == "x", "Annotation"] = "OutKB"

        return df


    def _run_regression_analysis(self, iaa_df):
        print("Fitting model for IAA of N={} datapoints.".format(len(iaa_df)))

        # 1. Fit the model.
        y = iaa_df["IAA"]
        X = iaa_df[["Suggestions", "Ambiguity"]]
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()

        print(model.summary())

        # 2. Actual vs Predicted Plot
        y_pred = model.predict(X)

        plt.figure(figsize=(6,5))
        plt.scatter(y, y_pred, alpha=0.7, color="blue")
        plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--')
        plt.xlabel("Actual IAA")
        plt.ylabel("Predicted IAA")
        plt.title("Actual vs Predicted IAA")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("actual_vs_predicted.png", dpi=300)
        plt.close()

        # === 3. Residual Plot ===
        residuals = model.resid

        plt.figure(figsize=(6,5))
        plt.scatter(y_pred, residuals, alpha=0.7, color="purple")
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted IAA")
        plt.ylabel("Residuals")
        plt.title("Residuals vs Predicted Values")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("residuals_plot.png", dpi=300)
        plt.close()

        # === 4. Partial Regression Plots ===
        fig = sm.graphics.plot_partregress_grid(model)
        fig.tight_layout()
        fig.savefig("partial_regression_plots.png", dpi=300)
        plt.close(fig)


    def _print_task_stats(self, annotations_df):
        results = defaultdict(lambda: defaultdict())

        acc, N = self._accuracy(self._task1_gold_df, annotations_df[annotations_df["Task"] == "Task 1"], task="Task 1")
        print("Task 1 Macro average accuracy over gold FrameIDs: {:.2f}".format(100*acc))
        results["Task 1"]["Acc"] = acc
        results["Task 1"]["N"] = int(N)

        for task2 in ["Task 2 High-Amb", "Task 2 Low-Amb"]:
            acc, N = self._accuracy(self._task2_gold_df, annotations_df[annotations_df["Task"] == task2], task=task2)
            print("{} Macro average accuracy over gold FrameIDs: {:.2f}".format(task2, 100*acc))
            results[task2]["Acc"] = acc
            results[task2]["N"] = int(N)

        acc, N = self._accuracy(self._task3_gold_df, annotations_df[annotations_df["Task"] == "Task 3"], task="Task 3")
        print("Task 3 Macro average accuracy over gold FrameIDs: {:.2f}".format(100*acc))
        results["Task 3"]["Acc"] = acc
        results["Task 3"]["N"] = int(N)

        for task in ["Task 1", "Task 2 High-Amb", "Task 2 Low-Amb", "Task 3"]:

            df = annotations_df[annotations_df["Task"] == task]
            N = len(df[df["Annotator"] == "1"])
            print("Task: {} (N={}), avg. num frames per lemma: {:.2f}, min: {}, max: {}.".format(task, N, np.mean(df["NumFrames"]), np.min(df["NumFrames"]), np.max(df["NumFrames"])))

            task_kappa, _, _ = self._fleiss_kappa_from_df(annotations_df[annotations_df["Task"] == task])
            print("Fleiss Kappa for Task: {} = {:.4f}".format(task, task_kappa))

            results[task]["IAA"] = task_kappa

        return results


    def _t_test(self, df, metric="IAA", task1="Task 1", task2="Task 2 High-Amb"):
        """Perform independent two-sample t-test."""

        print("----------")
        print(f"Welch's two-sample t-test on metric {metric} between tasks {task1} and {task2}:")

        task1_datapoints = df[df["Task"] == task1][metric]
        task2_datapoints = df[df["Task"] == task2][metric]

        t_stat, p_value = ttest_ind(task1_datapoints, task2_datapoints, equal_var=False)

        # Check significance
        alpha = 0.05
        if p_value < alpha:
            print(f"{metric} distributions are significantly different at alpha = {alpha} with p = {p_value:.5f}.")
        else:
            print(f"No significant difference in {metric} at alpha = {alpha} with p = {p_value:.5f}.")

        print("----------")


    def _accuracy(self, gold_df, pred_df, task):
        acc, N = [], []
        for annotator, annotator_pred_df in pred_df.groupby(by="Annotator"):
            merged = pd.merge(gold_df, annotator_pred_df, on="FrameID", how="inner", suffixes=("_gold", "_pred"))
            accuracy = (merged["Annotation_gold"] == merged["Annotation_pred"]).mean()
            acc.append(accuracy)
            N.append(len(merged))
        return np.mean(acc), np.mean(N)
