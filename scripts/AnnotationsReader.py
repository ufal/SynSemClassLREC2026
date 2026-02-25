#!/usr/bin/env python3

# Copyright 2025 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""AnnotationsReader."""


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


class AnnotationsReader():


    def __init__(self, excluded_filename=None, max_annotated_rows=None, verbose=1):
        self._max_annotated_rows = max_annotated_rows

        self._verbose = verbose

        # Read frames excluded later from annotation.
        if excluded_filename:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore",
                                        message="Data Validation extension is not supported and will be removed",
                                        category=UserWarning)
                self._excluded_df = pd.read_excel(excluded_filename, header=None, names=["Frame", "_", "Reason"])
        else:
            self._excluded_df = None


    def get_annotations(self):
        raise NotImplementedError("Subclasses of AnnotationsReader() must implement get_annotations().")


    def _read_xlsx(self, filename):
        if self._verbose >= 3:
            print("{}: Reading data from XLSX file {}".format(type(self).__name__,
                                                              filename))

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message="Data Validation extension is not supported and will be removed",
                                    category=UserWarning)
            return pd.read_excel(filename)


    def _exclude_frames(self, df):
        # Exclude lemma "to be" which is an outlier in all aspects.
        before = len(df)
        df = df[df["Lemma"] != "být"]
        after = len(df)
        if self._verbose >= 3:
            print("AnnotationsReader: Dropped {} frames of lemma \"to be\" of {} frames, {} remained.".format(before-after, before, after))

        if self._excluded_df is not None:
            before = len(df)
            df = df[~df["Frame"].isin(self._excluded_df["Frame"])]
            after = len(df)
            if self._verbose >= 3:
                print("AnnotationsReader: Dropped {} excluded frames of {} frames, {} remained.".format(before-after, before, after))
        return df


    def _drop_frames_without_gold_class(self, df, no_class_mark):
        before = len(df["Frame"].unique())

        # Extract frames without gold class because in Task 2 annotations, only
        # some of the rows are marked.
        frames_without_gold_class = df[df["Annotation"] == no_class_mark]["Frame"]

        df = df[~df["Frame"].isin(frames_without_gold_class)]

        after = len(df["Frame"].unique())

        if self._verbose >= 3:
            print("Dropped {} (of {}) frame(s) without gold class in SynSemClass, {} annotated frames remained.".format(before-after,
                                                                                                                        before,
                                                                                                                        after))
        return df


class AnnotationsReaderWithoutSuggestions(AnnotationsReader):


    def get_annotations(self, filename, drop_deleted=True, drop_frames_without_gold_class=True, first_class_only=False, num_frames_df=None, suggestion_type=None):
        """Reads annotations without suggestions from XLSX filename."""

        # 1. Read XLSX.
        df = self._read_xlsx(filename)

        # 2. Annotated rows cutoff.
        if self._max_annotated_rows:
            df = df.head(self._max_annotated_rows)

        # 3. Rename columns.
        df = df.rename(columns={"ID rámce": "Frame",
                                "Rámců v PDT_Vallexu": "NumFrames",
                                "1.třída – ID": "Annotation",
                                "2. třída – ID": "Annotation2",
                                "3. třída – ID": "Annotation3"})

        # 4. Drop empty rows and clean errors.
        df = df.dropna(subset=["Lemma"])
        df.loc[df["Annotation"] == "x ", "Annotation"] = "x"

        # 5. Exclude rejected frames.
        df = self._exclude_frames(df)

        # 6. Drop deleted frames.
        if drop_deleted:
            df = df[df["Annotation"] != "D"]

        # 7. Drop frames without SynSemClass class at the time of annotation.
        if drop_frames_without_gold_class:
            df = self._drop_frames_without_gold_class(df, "x")

        # 8. Drop lemmas outside the given number of frames.
        if num_frames_df is not None:
            df = df[df["Lemma"].isin(num_frames_df["Lemma"])]

        lemma2frames = defaultdict(lambda: set())
        annotations_dict = defaultdict(lambda: dict())
        for row in df.itertuples(index=False):
            lemma2frames[row.Lemma].add(row.Frame)
            annotations_dict[row.Frame][row.Annotation] = 1

            if first_class_only:
                continue

            if not pd.isna(row.Annotation2):
                annotations_dict[row.Frame][row.Annotation2] = 1
            if not pd.isna(row.Annotation3):
                annotations_dict[row.Frame][row.Annotation3] = 1

        return annotations_dict, lemma2frames


class AnnotationsReaderWithSuggestions(AnnotationsReader):


    def get_annotations_from_frame_group(self, annotation_dict, df, columns, label, value, first_class_only=False):
        """Reads Task 2 annotations per one frame."""

        if value == 0 and first_class_only:
            return annotation_dict

        class2value = dict()

        for _, row in df.iterrows():
            frame = row["Frame"]

            if value == 1:
                # Gold positive may be explicitly written in the "annotation" column.
                if row["annotation"] != "x" and not pd.isna(row["annotation"]):
                    annotation_dict[frame][row["annotation"]] = 1

                    # Explicitly annotated gold positive gets immediately returned.
                    if first_class_only:
                        return annotation_dict

            # Collect other annotations and their suggested values.
            cols_with_label = row[row == label].index.tolist()
            for col_with_label in cols_with_label:
                col_with_label = columns.get_loc(col_with_label)
                annotation_class = str(row.iloc[col_with_label-2]).split(" ")[0]
                annotation_value = str(row.iloc[col_with_label-1])
                class2value[annotation_class] = annotation_value

        # Find the first choice among gold positives.
        if value == 1 and first_class_only:
            if not class2value:
                return annotation_dict

            first_class = max(class2value, key=class2value.get)
            annotation_dict[frame][first_class] = 1
            return annotation_dict

        # Otherwise just return the annotated classes with the requested value.
        for annotated_class in class2value.keys():
            annotation_dict[frame][annotated_class] = value

        return annotation_dict


    def get_annotations(self, gold_filename, task="Task 2", suggestion_type=None, num_frames_df=None, drop_frames_without_gold_class=True, first_class_only=False, include_weak_annotations=True, drop_deleted=True):
        """Reads annotations with suggestions from XLSX filename."""

        # 1. Read XLSX file.
        df = self._read_xlsx(gold_filename)

        # 2. Annotated rows cutoff
        if self._max_annotated_rows:
            df = df.head(self._max_annotated_rows)

        # 3. Rename columns.
        df = df.rename(columns={"L(emma) or F(rame)": "SuggestionType"})
        df = df.replace("L", "lemma")
        df = df.replace("F", "frame")

        # 4. Drop empty rows and clean errors.
        df = df.dropna(subset=["Lemma"])
        df.loc[df["annotation"] == "x ", "annotation"] = "x"

        # 5. Exclude rejected frames.
        df = self._exclude_frames(df)

        # 6. Drop deleted frames.
        if drop_deleted:
            df = df[df["annotation"] != "D"]

        # 7. Drop frames without SynSemClass class at the time of annotation.
        # TODO: Use private function for this.
        if drop_frames_without_gold_class:
            num_gold_frames_before = len(df["Frame"].unique())
            frames_without_gold_class = df[df["annotation"] == 0]["Frame"].unique()
            df = df[~df["Frame"].isin(frames_without_gold_class)]
            num_gold_frames_after = len(df["Frame"].unique())

            if self._verbose >= 3:
                print("AnnotationsReaderWithSuggestions: Dropped {} (of {}) frame(s) without gold class in SynSemClass, {} annotated frames remained.".format(num_gold_frames_before-num_gold_frames_after,
                                                                                                                        num_gold_frames_before,
                                                                                                                        num_gold_frames_after))

        # 8. Drop lemmas outside the given number of frames.
        if num_frames_df is not None:
            df = df[df["Lemma"].isin(num_frames_df["Lemma"])]

        # Convert rather yes -> yes, rather no -> no.
        if include_weak_annotations:
            df = df.replace("r_y", "y")
            df = df.replace("r_n", "n")

        # Get annotated frames of suggested type.
        if suggestion_type:
            annotated_frames_of_type_df = df[(df["SuggestionType"] == suggestion_type) & (~pd.isna(df["class1"]))]
        else:
            annotated_frames_of_type_df = df[~pd.isna(df["class1"])]

        # Construct lemma2frames mapping.
        lemma2frames = defaultdict(lambda: set())
        for _, row in annotated_frames_of_type_df.iterrows():
            lemma2frames[row["Lemma"]].add(row["Frame"])

        # Get gold negative annotations only from rows with suggestion type annotation.
        gold_dict = defaultdict(lambda: dict())
        for _, group in annotated_frames_of_type_df.groupby("Frame"):
            gold_dict = self.get_annotations_from_frame_group(gold_dict,
                                                              group,
                                                              df.columns, "n", 0, first_class_only=first_class_only)

        # Get gold positive annotations from all rows with annotated frames.
        annotated_frames_complement_df = df[df["Frame"].isin(annotated_frames_of_type_df["Frame"])]
        for _, group in annotated_frames_complement_df.groupby("Frame"):
            gold_dict = self.get_annotations_from_frame_group(gold_dict,
                                                              group,
                                                              df.columns, "y", 1, first_class_only=first_class_only)

        if len(gold_dict.keys()) == 0:
            return None, None

        return gold_dict, lemma2frames


class GoldReader(AnnotationsReader):


    def get_annotations(self, filename, num_frames_df=None, suggestion_type="lemma", drop_deleted=True, drop_frames_without_gold_class=False):

        if self._max_annotated_rows:
            raise NotImplementedError("max_annotated_rows not impemented for GoldReader")

        if self._verbose >= 3:
            print("GoldReader: Reading gold data from file {}".format(filename))

        df = pd.read_csv(filename, sep="\t")
        df = df.rename(columns={"Finální třída": "Gold",
                                "Finalni trida": "Gold",
                                "ID ramce": "Frame"})

        # Clean errors
        df.loc[df["Gold"] == "x ", "Gold"] = "x"

        if drop_deleted:
            df = df[~df["Gold"].isin(["D", "P"])]

        # Drop unfinished

        # Drop frames excluded later from annotation.
        df = self._exclude_frames(df)

        # Drop frames without gold class
        if drop_frames_without_gold_class:
            df = df[df["Gold"] != "x"]

        # Drop empty rows and not yet annotated rows.
        df = df.dropna(subset=["Gold"])

        # Drop lemmas outside the given number of frames.
        if num_frames_df is not None:
            df = df[df["Lemma"].isin(num_frames_df["Lemma"])]

        # Construct the gold evaluation dict and lemma2frames mapping.
        annotations_dict = dict()
        lemma2frames = defaultdict(set)
        for _, row in df.iterrows():
            lemma2frames[row["Lemma"]].add(row["Frame"])
            if row["Frame"] not in annotations_dict.keys():
                annotations_dict[row["Frame"]] = dict()
            annotations_dict[row["Frame"]][row["Gold"]] = 1

        if len(annotations_dict.keys()) == 0:
            return None, lemma2frames

        return annotations_dict, lemma2frames
