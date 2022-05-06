# -*- coding: utf-8 -*-

import argparse
import glob
import logging
import os
import re
import json
from csv import reader
from pathlib import Path

import editdistance
import edlib
import termtables as tt

NOT_ENTITY_TAG = "O"

THRESHOLD = 0.30
BEGINNING_POS = ["B", "S", "U"]


def get_type_label(label: str) -> str:
    """Return the type (tag) of a label

    Input format: "[BIESLU]-type"
    """
    try:
        tag = (
            NOT_ENTITY_TAG
            if label == NOT_ENTITY_TAG
            else re.match(r"[BIESLU]-(.*)$", label)[1]
        )
    except TypeError:
        raise (Exception(f"The label {label} is not valid in BIOES/BIOLU format."))

    return tag


def get_position_label(label: str) -> str:
    """Return the position of a label

    Input format: "[BIESLU]-type"
    """
    try:
        pos = (
            NOT_ENTITY_TAG
            if label == NOT_ENTITY_TAG
            else re.match(r"([BIESLU])-(.*)$", label)[1]
        )
    except TypeError:
        raise (Exception(f"The label {label} is not valid in BIOES/BIOLU format."))

    return pos


def parse_bio(path: str) -> dict:
    """Parse a BIO file to get text content, character-level NE labels and entity types count.

    Input : path to a valid BIO file
    Output format : { "words": str, "labels": list; "entity_count" : { tag : int } }
    """
    assert os.path.exists(path), f"Error: Input file {path} does not exist"

    words = []
    labels = []
    entity_count = {"All": 0}
    last_tag = None

    with open(path, "r",encoding="utf-8") as fd:
        lines = list(filter(lambda x: x != "\n", fd.readlines()))

    if "§" in " ".join(lines):
        raise (
            Exception(
                f"§ found in input file {path}. Since this character is used in a specific way during evaluation, prease remove it from files."
            )
        )

    # Track nested entities infos
    in_nested_entity = False
    containing_tag = None

    for index, line in enumerate(lines):

        try:
            word, label = line.split()
        except ValueError:
            raise (
                Exception(
                    f"The file {path} given in input is not in BIO format: check line {index} ({line})"
                )
            )

        # Preserve hyphens to avoid confusion with the hyphens added later during alignment
        word = word.replace("-", "§")
        words.append(word)

        tag = get_type_label(label)

        # Spaces will be added between words and have to get a label
        if index != 0:

            # If new word has same tag as previous, not new entity and in entity, continue entity
            if (
                last_tag == tag
                and get_position_label(label) not in BEGINNING_POS
                and tag != NOT_ENTITY_TAG
            ):
                labels.append(f"I-{last_tag}")

            # If new word begins a new entity of different type, check for nested entity to correctly tag the space
            elif (
                last_tag != tag
                and get_position_label(label) in BEGINNING_POS
                and tag != NOT_ENTITY_TAG
                and last_tag != NOT_ENTITY_TAG
            ):

                # Advance to next word with different label as current
                future_label = label
                while (
                    index < len(lines)
                    and future_label != NOT_ENTITY_TAG
                    and get_type_label(future_label) != last_tag
                ):
                    index += 1
                    if index < len(lines):
                        future_label = lines[index].split()[1]

                # Check for continuation of the original entity
                if (
                    index < len(lines)
                    and get_position_label(future_label) not in BEGINNING_POS
                    and get_type_label(future_label) == last_tag
                ):
                    labels.append(f"I-{last_tag}")
                    in_nested_entity = True
                    containing_tag = last_tag
                else:
                    labels.append(NOT_ENTITY_TAG)
                    in_nested_entity = False

            elif in_nested_entity:
                labels.append(f"I-{containing_tag}")

            else:
                labels.append(NOT_ENTITY_TAG)
                in_nested_entity = False

        # Add a tag for each letter in the word
        if get_position_label(label) in BEGINNING_POS:
            labels += [f"B-{tag}"] + [f"I-{tag}"] * (len(word) - 1)
        else:
            labels += [label] * len(word)

        # Count nb entity for each type
        if get_position_label(label) in BEGINNING_POS:
            entity_count[tag] = entity_count.get(tag, 0) + 1
            entity_count["All"] += 1

        last_tag = tag

    result = None

    if words:

        result = dict()
        result["words"] = " ".join(words)
        result["labels"] = labels
        result["entity_count"] = entity_count

        assert len(result["words"]) == len(result["labels"])
        for tag in result["entity_count"]:
            if tag != "All":
                assert result["labels"].count(f"B-{tag}") == result["entity_count"][tag]

    return result


def look_for_further_entity_part(index, tag, characters, labels):
    """Get further entities parts for long entities with nested entities.

    Input:
        index: the starting index to look for rest of entity (one after last character included)
        tag: the type of the entity investigated
        characters: the string of the annotation or prediction
        the labels associated with characters
    Output :
        complete string of the rest of the entity found
        visited: indexes of the characters used for this last entity part OF THE DESIGNATED TAG. Do not process again later
    """
    original_index = index
    last_loop_index = index
    research = True
    visited = []
    while research:
        while (
            index < len(characters)
            and labels[index] != NOT_ENTITY_TAG
            and get_type_label(labels[index]) != tag
        ):
            index += 1
        while (
            index < len(characters)
            and get_position_label(labels[index]) not in BEGINNING_POS
            and get_type_label(labels[index]) == tag
        ):
            visited.append(index)
            index += 1

        research = index != last_loop_index and get_type_label(labels[index - 1]) == tag
        last_loop_index = index

    characters_to_add = (
        characters[original_index:index]
        if get_type_label(labels[index - 1]) == tag
        else []
    )

    return characters_to_add, visited


def compute_matches(
    annotation: str,
    prediction: str,
    labels_annot: list,
    labels_predict: list,
    threshold: int,
) -> dict:
    """Compute prediction score from annotation string to prediction string.

    Annotation and prediction strings should be the same length.

    For each entity in the annotation string, a match is found in the prediction.
    This is done in looking for a sub-string roughly at the same position in the prediction, and with the right entity-tag.
    Here is an example to illustrate the method used :

                     *-------*       *----*
    labels_annot   : PPPPPPPPPOOOOOOOCCCCCCOO
    annotation     : Tolkie-n- was a writer .
    prediction     : Tolkieene xas --writer .
    labels_predict : PPPPPPPPPOCCCCCCCCCCCCCC
                     *-------* <-----*----*->

    Each entity in the annotation string gets a prediction score based on the number
    of characters well predicted and labeled in the prediction string.
    The score of a label is the addition of entity scores divided by the number
    of entities.

    Inputs :
    annotation : str, example : "Tolkie-n- was a writer- -."
    prediction : str, example : "Tolkieene xas --writear ,."
    labels_annot : list of strings,   example : ['B-P','I-P','I-P','I-P','I-P','I-P','I-P','I-P','I-P','O', ...]
    labels_predict : list of string , example : ['B-P','I-P','I-P','I-P','I-P','I-P','I-P','I-P','I-P','O', ...]

    Output : {TAG1 : nb_entity_matched, ...}, example : {'All': 1, 'OCC': 0, 'PER': 1}
    """
    assert annotation
    assert prediction
    assert labels_annot
    assert labels_predict

    entity_count = {"All": 0}
    last_tag = NOT_ENTITY_TAG

    # Track indexes of characters found for continuation of nested entities
    visited_annot = []
    visited_predict = []

    # Iterating on reference string
    for i, char_annot in enumerate(annotation):

        if i in visited_annot:
            continue

        label_ref = labels_annot[i]
        tag_ref = get_type_label(label_ref)
        label_predict = labels_predict[i]
        tag_predict = get_type_label(label_predict)

        # If character not in entity
        if tag_ref == NOT_ENTITY_TAG:
            last_tag = NOT_ENTITY_TAG

        else:

            # If beginning new entity
            if get_position_label(label_ref) in BEGINNING_POS:
                current_ref, current_compar = [], []
                last_tag = tag_ref
                found_aligned_beginning = False
                found_aligned_end = False

            current_ref.append(char_annot)

            # Searching character string corresponding with tag
            if not found_aligned_end and tag_predict == tag_ref:

                if i in visited_predict:
                    continue

                # If just beginning new entity, backtrack tags on prediction string
                if (
                    len(current_ref) == 1
                    and get_position_label(labels_predict[i]) not in BEGINNING_POS
                ):
                    j = i - 1
                    while (
                        j >= 0
                        and get_type_label(labels_predict[j]) == tag_ref
                        and get_position_label(labels_predict[j]) not in BEGINNING_POS
                        and j not in visited_predict
                    ):
                        j -= 1

                    if (
                        get_position_label(labels_predict[j]) in BEGINNING_POS
                        and get_type_label(labels_predict[j]) == tag_ref
                        and j not in visited_predict
                    ):
                        start = j
                    else:
                        start = j + 1

                    current_compar += prediction[start:i]

                found_aligned_beginning = True
                current_compar.append(prediction[i])

            # If tags don't match and beginning was found : end of predicted entity
            elif found_aligned_beginning:
                found_aligned_end = True

            # If detect end of (1st part) entity in annotation: check for nested entity and compare
            if (i + 1 == len(annotation)) or (
                i + 1 < len(annotation)
                and get_type_label(labels_annot[i + 1]) != last_tag
            ):

                if not found_aligned_end:
                    rest_predict, visited = look_for_further_entity_part(
                        i + 1, tag_ref, prediction, labels_predict
                    )
                    current_compar += rest_predict
                    visited_predict += visited

                rest_annot, visited = look_for_further_entity_part(
                    i + 1, tag_ref, annotation, labels_annot
                )
                current_ref += rest_annot
                visited_annot += visited

                # Normalize collected strings
                entity_ref = "".join(current_ref)
                entity_ref = entity_ref.replace("-", "")
                len_entity = len(entity_ref)
                entity_compar = "".join(current_compar)
                entity_compar = entity_compar.replace("-", "")

                # One entity is counted as recognized (score of 1) if the Levenhstein distance between the expected and predicted entities
                # represents less than 30% (THRESHOLD) of the length of the expected entity.
                # Precision and recall will be computed for each category in comparing the numbers of recognized entities and expected entities
                score = (
                    1
                    if editdistance.eval(entity_ref, entity_compar) / len_entity
                    <= threshold
                    else 0
                )
                entity_count[last_tag] = entity_count.get(last_tag, 0) + score
                entity_count["All"] += score
                current_ref = []
                current_compar = []

    return entity_count


def get_labels_aligned(original: str, aligned: str, labels_original: list) -> list:
    """Takes original string, original string labels and aligned string given by edlib.align.
    Returns a list of labels corresponding to the aligned string.

    Input formats:
        original: str
        aligned: str with hyphens
        labels_original: list of labels ["O", "B-LOC", "I-LOC", ...]
    Output format :
        list of strings
    """
    assert original
    assert aligned
    assert labels_original

    labels_aligned = []
    index_original = 0
    last_label = NOT_ENTITY_TAG

    # Inspecting aligned string
    for i, char in enumerate(aligned):
        # new_label = ""

        # If original string has been fully processed, rest of labels are "O" ('-' characters at aligned end)
        if index_original >= len(original):
            new_label = NOT_ENTITY_TAG

        # If current aligned char does not match current original char ('-' characters in aligned)
        # Keep last_label and don't increment index_original
        elif not char == original[index_original]:
            new_label = (
                last_label
                if get_position_label(last_label) not in BEGINNING_POS
                else f"I-{get_type_label(last_label)}"
            )

        # Until matching of characters)
        else:
            new_label = labels_original[index_original]
            last_label = new_label
            index_original += 1

        labels_aligned.append(new_label)

    return labels_aligned


def compute_scores(
    annot_tags_count: dict, predict_tags_count: dict, matches: dict
) -> dict:
    """Compute Precision, Recall and F1 score for all entity types found in annotation and prediction.

    Each measure is given at document level, global score is a micro-average over tag types.

    Inputs :
    annot :   { TAG1(str) : nb_entity(int), ...}
    predict : { TAG1(str) : nb_entity(int), ...}
    matches : { TAG1(str) : nb_entity_matched(int), ...}

    Output :
    scores : { TAG1(str) : {"P" : float, "R" : float, "F1" : float}, ... }
    """

    annot_tags = set(annot_tags_count.keys())
    predict_tags = set(predict_tags_count.keys())
    tags = annot_tags | predict_tags

    scores = {tag: {"P": None, "R": None, "F1": None} for tag in tags}

    for tag in sorted(tags)[::-1]:
        nb_predict = predict_tags_count.get(tag)
        nb_annot = annot_tags_count.get(tag)
        nb_match = matches.get(tag, 0)
        prec = None if not nb_predict else nb_match / nb_predict
        rec = None if not nb_annot else nb_match / nb_annot
        f1 = (
            None
            if (prec is None) or (rec is None)
            else 0
            if (prec + rec == 0)
            else 2 * (prec * rec) / (prec + rec)
        )

        scores[tag]["predicted"] = nb_predict
        scores[tag]["matched"] = nb_match
        scores[tag]["P"] = prec
        scores[tag]["R"] = rec
        scores[tag]["F1"] = f1
        scores[tag]["Support"] = nb_annot

    return scores


def print_results(scores: dict):
    """Display final results.

    None values are kept to indicate the absence of a certain tag in either annotation or prediction.
    """
    header = ["tag", "predicted", "matched", "Precision", "Recall", "F1", "Support"]
    results = []
    for tag in sorted(scores.keys())[::-1]:
        prec = None if scores[tag]["P"] is None else round(scores[tag]["P"], 3)
        rec = None if scores[tag]["R"] is None else round(scores[tag]["R"], 3)
        f1 = None if scores[tag]["F1"] is None else round(scores[tag]["F1"], 3)

        results.append(
            [
                tag,
                scores[tag]["predicted"],
                scores[tag]["matched"],
                prec,
                rec,
                f1,
                scores[tag]["Support"],
            ]
        )
    tt.print(results, header, style=tt.styles.markdown)


def print_result_compact(scores: dict):
    result = []
    header = ["tag", "predicted", "matched", "Precision", "Recall", "F1", "Support"]
    result.append(
        [
            "ALl",
            scores["All"]["predicted"],
            scores["All"]["matched"],
            round(scores["All"]["P"], 3),
            round(scores["All"]["R"], 3),
            round(scores["All"]["F1"], 3),
            scores["All"]["Support"],
        ]
    )
    tt.print(result, header, style=tt.styles.markdown)


def run(annotation: str, prediction: str, threshold: int, verbose: bool) -> dict:
    """Compute recall and precision for each entity type found in annotation and/or prediction.

    Each measure is given at document level, global score is a micro-average across entity types.
    """
    # Get string and list of labels per character
    annot = parse_bio(annotation)
    predict = parse_bio(prediction)

    if not annot or not predict:
        raise Exception("No content found in annotation or prediction files.")

    # Align annotation and prediction
    align_result = edlib.align(annot["words"], predict["words"], task="path")
    nice_alignment = edlib.getNiceAlignment(
        align_result, annot["words"], predict["words"]
    )

    annot_aligned = nice_alignment["query_aligned"]
    predict_aligned = nice_alignment["target_aligned"]

    # Align labels from string alignment
    labels_annot_aligned = get_labels_aligned(
        annot["words"], annot_aligned, annot["labels"]
    )
    labels_predict_aligned = get_labels_aligned(
        predict["words"], predict_aligned, predict["labels"]
    )

    # Get nb match
    matches = compute_matches(
        annot_aligned,
        predict_aligned,
        labels_annot_aligned,
        labels_predict_aligned,
        threshold,
    )

    # Compute scores
    scores = compute_scores(annot["entity_count"], predict["entity_count"], matches)

    # Sorties d'alignements
    string_bio = ""
    for i,char in enumerate(annot_aligned):
        if char != "-":
            #string_bio += char+" "+labels_annot_aligned[i]+" "+labels_predict_aligned[i]+"\n"
            string_bio += char+" "+predict_aligned[i]+" "+labels_annot_aligned[i]+" "+labels_predict_aligned[i]+"\n"
    with open("aligned.txt","w",encoding="utf-8") as f:
        f.write(string_bio)
    #with open("align_loc_pp.json","w") as f:
        #f.write(json.dumps(labels_annot_aligned,indent=2,ensure_ascii=False))
    #with open("align_loc_hyp.json","w") as f:
        #f.write(json.dumps(labels_predict_aligned,indent=2,ensure_ascii=False))

    # Print results
    if verbose:
        print_results(scores)
    else:
        print_result_compact(scores)

    return scores


def run_multiple(file_csv, folder, threshold, verbose):
    """Run the program for multiple files (correlation indicated in the csv file)"""
    # Read the csv in a list
    with open(file_csv, "r") as read_obj:
        csv_reader = reader(read_obj)
        list_cor = list(csv_reader)

    if os.path.isdir(folder):
        list_bio_file = glob.glob(str(folder) + "/**/*.bio", recursive=True)

        count = 0
        precision = 0
        recall = 0
        f1 = 0
        for row in list_cor:
            annot = None
            predict = None

            for file in list_bio_file:
                if row[0] == os.path.basename(file):
                    annot = file
            for file in list_bio_file:
                if row[1] == os.path.basename(file):
                    predict = file

            if annot and predict:
                count += 1
                print(os.path.basename(predict))
                scores = run(annot, predict, threshold, verbose)
                precision += scores["All"]["P"]
                recall += scores["All"]["R"]
                f1 += scores["All"]["F1"]
                print()
            else:
                raise Exception(f"No file found for files {annot}, {predict}")
        if count:
            print("Average score on all corpus")
            tt.print(
                [
                    [
                        round(precision / count, 3),
                        round(recall / count, 3),
                        round(f1 / count, 3),
                    ]
                ],
                ["Precision", "Recall", "F1"],
                style=tt.styles.markdown,
            )
        else:
            raise Exception("No file were counted")
    else:
        raise Exception("the path indicated does not lead to a folder.")


def threshold_float_type(arg):
    """Type function for argparse."""
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number.")
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError("Must be between 0 and 1.")
    return f


def main():
    """Get arguments and run."""

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Compute score of NER on predict.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-a",
        "--annot",
        help="Annotation in BIO format.",
    )
    group.add_argument(
        "-c",
        "--csv",
        help="Csv with the correlation between the annotation bio files and the predict bio files",
        type=Path,
    )
    parser.add_argument(
        "-p",
        "--predict",
        help="Prediction in BIO format.",
    )
    parser.add_argument(
        "-f",
        "--folder",
        help="Folder containing the bio files referred to in the csv file",
        type=Path,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Print only the recap if False",
        action="store_false",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        help="Set a distance threshold for the match between gold and predicted entity.",
        default=THRESHOLD,
        type=threshold_float_type,
    )

    args = parser.parse_args()

    if args.annot:
        if not args.predict:
            raise parser.error("You need to specify the path to a predict file with -p")
        if args.annot and args.predict:
            run(args.annot, args.predict, args.threshold, args.verbose)
    elif args.csv:
        if not args.folder:
            raise parser.error(
                "You need to specify the path to a folder of bio files with -f"
            )
        if args.folder and args.csv:
            run_multiple(args.csv, args.folder, args.threshold, args.verbose)
    else:
        raise parser.error("You need to specify the argument of input file")


if __name__ == "__main__":
    main()
