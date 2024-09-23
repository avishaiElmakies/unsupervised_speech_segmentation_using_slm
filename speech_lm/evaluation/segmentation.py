from typing import Any,Mapping,List
import torch
from pyannote.metrics.segmentation import (SegmentationPurityCoverageFMeasure,
                                    SegmentationCoverage,
                                    SegmentationPurity,SegmentationRecall,SegmentationPrecision)
from pyannote.metrics.base import f_measure
from pyannote.core import Segment, Timeline
from math import sqrt
import scipy.stats

COVERAGE = "coverage"
PURITY = "purity"
CPFMEASURE = "cpfmeasure"
RECALL = "recall"
PRECISION = "precision"
RPFMEASURE = "rpfmeasure"
R_VALUE = "r_value"

AVAILABLE_METRICS = [COVERAGE,PURITY, CPFMEASURE, RECALL, PRECISION,RPFMEASURE, R_VALUE]

EPS = 1e-10

TOL = 0.5

def r_value(precision,recall):
    """
    calc rvalue from precision and recall
    """
    os = recall / (precision + EPS) - 1
    r1 = sqrt((1 - recall) ** 2 + os ** 2)
    r2 = (-os + recall - 1) / (sqrt(2))
    rval = 1 - (abs(r1) + abs(r2)) / 2
    return rval

def get_metric(name: str):
    """
    get metric by name
    """
    if name == COVERAGE:
        return SegmentationCoverage()
    elif name == PURITY:
        return SegmentationPurity()
    elif name == CPFMEASURE:
        return SegmentationPurityCoverageFMeasure()
    elif name == RECALL:
        return SegmentationRecall(tolerance=TOL)
    elif name == PRECISION:
        return SegmentationPrecision(tolerance=TOL)
    elif name == RPFMEASURE:
        return SegmentationPrecisionRecallFMeasure(tolerance=TOL)
    elif name == R_VALUE:
        return SegmentationPrecisionRecallRValue(tolerance=TOL)
    else:
        raise ValueError(f"Metric {name} not available")
    

def to_timeline(segments: List[Mapping[str,float]]):
    """
    convert list of segments to timeline
    """
    return Timeline([Segment(seg["start"], seg["end"]) for seg in segments])

def to_annotation(segments: List[Mapping[str,float]]):
    """
    convert list of segments to annotation
    """
    return to_timeline(segments).to_annotation()


class SegmentationPrecisionRecallRValue:

    """
    Segmentation precision recall r value
    """

    def __init__(self, tolerance: float = 0.5) -> None:
        self.tolerance = tolerance
        self.precision = SegmentationPrecision(tolerance=self.tolerance)
        self.recall = SegmentationRecall(tolerance=self.tolerance)
        self.results_ = []

    def __call__(self, reference,hypothesis,detailed=False) -> Any:
        """
        get precision recall r value for single sample
        """
        precision = self.precision(reference,hypothesis)
        recall = self.recall(reference,hypothesis)
        rval = r_value(precision,recall)
        self.results_.append(rval)
        if detailed:
            return {"precision": precision, "recall": recall, "rval": rval}
        return rval
    
    def __abs__(self):
        presicion = abs(self.precision)
        recall = abs(self.recall)
        return r_value(presicion,recall)

    def confidence_interval(self, alpha=0.9):
        """
        get confidence interval for segmentation
        """
        return scipy.stats.bayes_mvs(self.results_, alpha=alpha)[0]


class SegmentationPrecisionRecallFMeasure:

    """
    Segmentation precision recall f measure
    """

    def __init__(self, tolerance: float = 0.5, beta=1) -> None:
        self.tolerance = tolerance
        self.beta = beta
        self.precision = SegmentationPrecision(tolerance=self.tolerance)
        self.recall = SegmentationRecall(tolerance=self.tolerance) 
        self.results_ = [] 

    def __call__(self, reference,hypothesis,detailed=False) -> Any:
        """
        get precision recall f measure for single sample
        """
        results = {}
        results["precision"] = self.precision(reference,hypothesis,detailed=detailed)
        results["recall"] = self.recall(reference,hypothesis,detailed=detailed)
        results["f_measure"] = f_measure(results["precision"],results["recall"],beta=self.beta)
        self.results_.append(results["f_measure"])
        if detailed:
            return results
        return results["f_measure"]
    
    def __abs__(self):
        presicion = abs(self.precision)
        recall = abs(self.recall)
        return f_measure(presicion,recall,beta=self.beta)

    def confidence_interval(self, alpha=0.9):
        """
        get confidence interval for segmentation
        """
        return scipy.stats.bayes_mvs(self.results_, alpha=alpha)[0]


class SegmentationMetrics:

    """
    Segmentation metrics, class handle all metrics at once
    """

    def __init__(self, metrics) -> None:
        if not all(metric in AVAILABLE_METRICS for metric in metrics):
            raise ValueError(f"Metrics should be one of {AVAILABLE_METRICS}")
        if len(metrics) == 0:
            raise ValueError("At least one metric should be provided")
        if (COVERAGE in metrics or PURITY in metrics) and CPFMEASURE in metrics:
            Warning("using CPFMEASURE with COVERAGE or PURITY is not recommended")
        self.metrics = {metric: get_metric(metric) for metric in metrics}

    
    def __call__(self, reference, hypothesis,detailed=False) -> Any:
        """
        get metric for all metrics
        """
        results = {}
        for metric in self.metrics:
            if not detailed:
                results[metric] = self.metrics[metric](reference,hypothesis)
            else:
                results.update(self.metrics[metric](reference,hypothesis,detailed=detailed))
        return results
    
    def __abs__(self):
        results = {}
        for metric in self.metrics:
            results[metric] = abs(self.metrics[metric])
        return results

    def confidence_interval(self, alpha=0.9):
        """
        get confidence interval for all metrics
        """
        results = {}
        for metric in self.metrics:
            results[metric] = self.metrics[metric].confidence_interval(alpha=alpha)
        return results
        
        
