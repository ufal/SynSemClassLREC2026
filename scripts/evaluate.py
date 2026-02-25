#!/usr/bin/env python3

# Copyright 2025 Institute of Formal and Applied Linguistics, Faculty of
# Mathematics and Physics, Charles University in Prague, Czech Republic.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.

"""SynSemClass automatic annotation suggestions evaluation.

Evaluates automatic suggestions against gold manual annotations.

The inputs into the annotation pipeline were valency frames (i.e., verb senses)
and the goal was to sort the input frames into the SynSemClass ontology
semantic classes. The annotators can either process the inputs manually without
any assistance (Task 1) or they receive K (K=5) automatically retrieved
semantic class suggestions for each of the input frame.

The script computes various metrics to assess the efficiency of the automatic
suggestions to answer several research questions.
"""


import argparse
import sys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=["task1", "task2"], default="task2", help="Task 1 (wo/ retrieved suggestions) or Task 2 (w/ retrieved suggestions).")
    args = parser.parse_args()
