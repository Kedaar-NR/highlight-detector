"""
Evaluation metrics and testing framework for highlight detection.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class GroundTruthEvent:
    """Ground truth highlight event."""
    start_time: float
    end_time: float
    label: str
    category: str
    confidence: float = 1.0


@dataclass
class DetectedEvent:
    """Detected highlight event."""
    start_time: float
    end_time: float
    label: str
    category: str
    confidence: float


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for highlight detection."""
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    average_precision: float
    mean_absolute_error: float
    detection_latency: float


class HighlightEvaluator:
    """Evaluates highlight detection performance."""
    
    def __init__(self, tolerance_window: float = 2.0):
        self.tolerance_window = tolerance_window
    
    def evaluate(
        self,
        ground_truth: List[GroundTruthEvent],
        detected: List[DetectedEvent],
        total_duration: float
    ) -> EvaluationMetrics:
        """Evaluate detection performance against ground truth."""
        
        # Calculate basic metrics
        tp, fp, fn, tn = self._calculate_confusion_matrix(
            ground_truth, detected, total_duration
        )
        
        # Calculate precision, recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Calculate accuracy
        total_events = tp + fp + fn + tn
        accuracy = (tp + tn) / total_events if total_events > 0 else 0.0
        
        # Calculate average precision
        ap = self._calculate_average_precision(ground_truth, detected)
        
        # Calculate mean absolute error for timing
        mae = self._calculate_timing_error(ground_truth, detected)
        
        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy=accuracy,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            true_negatives=tn,
            average_precision=ap,
            mean_absolute_error=mae,
            detection_latency=0.0  # Will be set separately
        )
    
    def _calculate_confusion_matrix(
        self,
        ground_truth: List[GroundTruthEvent],
        detected: List[DetectedEvent],
        total_duration: float
    ) -> Tuple[int, int, int, int]:
        """Calculate confusion matrix for detection evaluation."""
        
        # Create time windows for ground truth events
        gt_windows = []
        for event in ground_truth:
            start = max(0, event.start_time - self.tolerance_window)
            end = min(total_duration, event.end_time + self.tolerance_window)
            gt_windows.append((start, end, event))
        
        # Track matches
        gt_matched = [False] * len(ground_truth)
        detected_matched = [False] * len(detected)
        
        # Find matches
        for i, detected_event in enumerate(detected):
            for j, (gt_start, gt_end, gt_event) in enumerate(gt_windows):
                if not gt_matched[j] and not detected_matched[i]:
                    # Check if detected event overlaps with ground truth window
                    if (detected_event.start_time < gt_end and 
                        detected_event.end_time > gt_start):
                        gt_matched[j] = True
                        detected_matched[i] = True
                        break
        
        # Calculate confusion matrix
        tp = sum(gt_matched)  # True positives
        fp = len(detected) - tp  # False positives
        fn = len(ground_truth) - tp  # False negatives
        
        # True negatives (no events in time periods without ground truth)
        # This is approximated as the total duration minus all event periods
        total_event_time = sum(
            (event.end_time - event.start_time) for event in ground_truth
        )
        tn = int((total_duration - total_event_time) / self.tolerance_window)
        
        return tp, fp, fn, tn
    
    def _calculate_average_precision(
        self,
        ground_truth: List[GroundTruthEvent],
        detected: List[DetectedEvent]
    ) -> float:
        """Calculate average precision for detection."""
        if not detected:
            return 0.0
        
        # Sort detected events by confidence
        sorted_detected = sorted(detected, key=lambda x: x.confidence, reverse=True)
        
        # Calculate precision at each recall level
        precisions = []
        recalls = []
        
        for i in range(1, len(sorted_detected) + 1):
            # Get top i detections
            top_i = sorted_detected[:i]
            
            # Calculate precision and recall
            tp = self._count_true_positives(ground_truth, top_i)
            precision = tp / i
            recall = tp / len(ground_truth) if ground_truth else 0.0
            
            precisions.append(precision)
            recalls.append(recall)
        
        # Calculate average precision using 11-point interpolation
        ap = 0.0
        for recall_threshold in np.arange(0, 1.1, 0.1):
            # Find maximum precision at this recall level
            max_precision = 0.0
            for i, recall in enumerate(recalls):
                if recall >= recall_threshold:
                    max_precision = max(max_precision, precisions[i])
            ap += max_precision
        
        return ap / 11.0
    
    def _count_true_positives(
        self,
        ground_truth: List[GroundTruthEvent],
        detected: List[DetectedEvent]
    ) -> int:
        """Count true positives between ground truth and detected events."""
        tp = 0
        gt_matched = [False] * len(ground_truth)
        
        for detected_event in detected:
            for i, gt_event in enumerate(ground_truth):
                if not gt_matched[i]:
                    # Check if events overlap within tolerance
                    if (detected_event.start_time < gt_event.end_time + self.tolerance_window and
                        detected_event.end_time > gt_event.start_time - self.tolerance_window):
                        gt_matched[i] = True
                        tp += 1
                        break
        
        return tp
    
    def _calculate_timing_error(
        self,
        ground_truth: List[GroundTruthEvent],
        detected: List[DetectedEvent]
    ) -> float:
        """Calculate mean absolute error for event timing."""
        if not detected or not ground_truth:
            return 0.0
        
        errors = []
        
        for detected_event in detected:
            # Find closest ground truth event
            min_error = float('inf')
            for gt_event in ground_truth:
                # Calculate timing error (center point difference)
                detected_center = (detected_event.start_time + detected_event.end_time) / 2
                gt_center = (gt_event.start_time + gt_event.end_time) / 2
                error = abs(detected_center - gt_center)
                min_error = min(min_error, error)
            
            if min_error != float('inf'):
                errors.append(min_error)
        
        return np.mean(errors) if errors else 0.0


class PerformanceBenchmark:
    """Benchmarks detection and rendering performance."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_detection(
        self,
        video_path: str,
        mode: str,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Benchmark detection performance."""
        logger.info(f"Benchmarking detection for {video_path} ({mode} mode)")
        
        times = []
        memory_usage = []
        
        for i in range(iterations):
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # This would call the actual detection pipeline
            # For now, we'll simulate it
            time.sleep(1)  # Simulate processing
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        return {
            "average_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "average_memory": np.mean(memory_usage),
            "video_duration": self._get_video_duration(video_path),
            "iterations": iterations
        }
    
    def benchmark_rendering(
        self,
        input_path: str,
        output_path: str,
        preset: str,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """Benchmark rendering performance."""
        logger.info(f"Benchmarking rendering for {input_path} -> {output_path}")
        
        times = []
        
        for i in range(iterations):
            start_time = time.time()
            
            # This would call the actual rendering pipeline
            # For now, we'll simulate it
            time.sleep(0.5)  # Simulate rendering
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        return {
            "average_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "preset": preset,
            "iterations": iterations
        }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return 0.0
    
    def _get_video_duration(self, video_path: str) -> float:
        """Get video duration in seconds."""
        try:
            import ffmpeg
            probe = ffmpeg.probe(video_path)
            return float(probe['format']['duration'])
        except Exception:
            return 0.0


class GoldenTestSuite:
    """Golden test suite for regression testing."""
    
    def __init__(self, test_data_dir: str = "./data/fixtures"):
        self.test_data_dir = Path(test_data_dir)
        self.test_data_dir.mkdir(parents=True, exist_ok=True)
    
    def create_golden_test(
        self,
        test_name: str,
        input_video: str,
        expected_events: List[GroundTruthEvent],
        mode: str = "sports"
    ) -> Dict[str, Any]:
        """Create a golden test case."""
        test_data = {
            "test_name": test_name,
            "input_video": input_video,
            "mode": mode,
            "expected_events": [
                {
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "label": event.label,
                    "category": event.category,
                    "confidence": event.confidence
                }
                for event in expected_events
            ],
            "created_at": time.time()
        }
        
        test_file = self.test_data_dir / f"{test_name}.json"
        with open(test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        logger.info(f"Created golden test: {test_name}")
        return test_data
    
    def run_golden_test(
        self,
        test_name: str,
        detector_func: callable
    ) -> Dict[str, Any]:
        """Run a golden test and compare results."""
        test_file = self.test_data_dir / f"{test_name}.json"
        
        if not test_file.exists():
            return {
                "success": False,
                "error": f"Test file not found: {test_file}"
            }
        
        # Load test data
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        # Run detection
        try:
            detected_events = detector_func(
                test_data["input_video"],
                test_data["mode"]
            )
            
            # Convert to DetectedEvent objects
            detected = [
                DetectedEvent(
                    start_time=event["start_time"],
                    end_time=event["end_time"],
                    label=event["label"],
                    category=event["category"],
                    confidence=event["confidence"]
                )
                for event in detected_events
            ]
            
            # Convert expected events
            expected = [
                GroundTruthEvent(
                    start_time=event["start_time"],
                    end_time=event["end_time"],
                    label=event["label"],
                    category=event["category"],
                    confidence=event["confidence"]
                )
                for event in test_data["expected_events"]
            ]
            
            # Evaluate
            evaluator = HighlightEvaluator()
            metrics = evaluator.evaluate(expected, detected, 300.0)  # Assume 5 min video
            
            return {
                "success": True,
                "test_name": test_name,
                "metrics": {
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1_score": metrics.f1_score,
                    "accuracy": metrics.accuracy
                },
                "detected_count": len(detected),
                "expected_count": len(expected)
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_all_tests(self, detector_func: callable) -> Dict[str, Any]:
        """Run all golden tests."""
        results = {}
        
        for test_file in self.test_data_dir.glob("*.json"):
            test_name = test_file.stem
            result = self.run_golden_test(test_name, detector_func)
            results[test_name] = result
        
        # Calculate overall metrics
        successful_tests = [r for r in results.values() if r.get("success", False)]
        
        if successful_tests:
            avg_precision = np.mean([r["metrics"]["precision"] for r in successful_tests])
            avg_recall = np.mean([r["metrics"]["recall"] for r in successful_tests])
            avg_f1 = np.mean([r["metrics"]["f1_score"] for r in successful_tests])
            avg_accuracy = np.mean([r["metrics"]["accuracy"] for r in successful_tests])
        else:
            avg_precision = avg_recall = avg_f1 = avg_accuracy = 0.0
        
        return {
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(results) - len(successful_tests),
            "average_metrics": {
                "precision": avg_precision,
                "recall": avg_recall,
                "f1_score": avg_f1,
                "accuracy": avg_accuracy
            },
            "test_results": results
        }
