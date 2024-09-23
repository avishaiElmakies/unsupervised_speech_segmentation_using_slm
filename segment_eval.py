import argparse
from speech_lm.evaluation.segmentation import SegmentationMetrics, to_annotation, to_timeline, AVAILABLE_METRICS
import json




def main():
    parser = argparse.ArgumentParser("this script evaluates segmented audio files segmentation."\
                                     "should use a json file you should have a dict where each key contains another dict with a key for segmentation" \
                                    "with a list of segments. each segment should have a start and end time", "multiple files can be evaluated at once if they are in the same json")
    parser.add_argument('-re','--reference_path', type=str, help='Path to the json file that contains the reference segmentation',required=True)
    parser.add_argument('-hy','--hypothesis_path', type=str, help='Path to the json file that contains the predicted segmentation',required=True)
    parser.add_argument('-m','--metrics', type=str,nargs="+", help='metrics to use for evaluation',choices=AVAILABLE_METRICS + ["all"],required=True)
    parser.add_argument('-p','--print_sub', action="store_true", help='print the results')
    parser.add_argument('-o','--output', type=str,default=None, help='Path to the output file if None will print to stdout')
    parser.add_argument('-ci','--confidence_interval', action="store_true", help='add confidence interval')

    args = parser.parse_args()
    reference_path  = args.reference_path
    with open(reference_path) as f:
        reference = json.load(f)
    hypothesis_path = args.hypothesis_path
    with open(hypothesis_path) as f:
        hypothesis = json.load(f)

    print_sub = args.print_sub

    m = args.metrics
    if "all" in m:
       m = AVAILABLE_METRICS

    metrics = SegmentationMetrics(m)

    for key,referene_values in reference.items():
        if key not in hypothesis:
            continue
        hypothesis_values = hypothesis[key]
        for sub_key,reference_segments in referene_values.items():
            if sub_key not in hypothesis_values:
                continue
            hypothesis_segments = hypothesis_values[sub_key]["segmentation"]
            reference_segments = reference_segments["segmentation"]
            reference_annotation = to_annotation(reference_segments)
            hypothesis_annotation = to_timeline(hypothesis_segments)
            results = metrics(reference_annotation,hypothesis_annotation)
            if print_sub:
                print(f"Results for {key} {sub_key}:")
                for metric in results:
                    print(f"{metric}: {results[metric]}")
                print("")

    results = abs(metrics)
    if args.confidence_interval:
        results["confidence_interval"] = metrics.confidence_interval()
    if args.output:
        with open(args.output,"w") as f:
            json.dump(results,f,indent=4)
    else:
        print(results)

    


if __name__ == '__main__':
    main()