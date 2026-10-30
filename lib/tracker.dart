import 'dart:ui' as ui;

import 'detector.dart';

/// Steadies live detections across frames so boxes don't flicker.
///
/// A box is shown once it has been seen on [minHits] frames, then kept until
/// it has been missing for more than [maxMisses] frames in a row. New boxes
/// need [showConfidence]; boxes already shown are kept down to
/// [keepConfidence]. Matched boxes are smoothed toward the latest position.
class DetectionTracker {
  DetectionTracker({
    this.showConfidence = 0.5,
    this.keepConfidence = 0.3,
    this.minHits = 2,
    this.maxMisses = 2,
    this.matchIou = 0.3,
    this.smoothing = 0.5,
  });

  final double showConfidence;
  final double keepConfidence;
  final int minHits;
  final int maxMisses;
  final double matchIou;

  /// Weight of the newest box when blending with the tracked one (0-1).
  final double smoothing;

  final List<_Track> _tracks = [];

  /// Feeds one frame's detections and returns the boxes to display.
  List<Detection> update(List<Detection> detections) {
    final unmatched = _tracks.toSet();
    final sorted = [...detections]
      ..sort((a, b) => b.confidence.compareTo(a.confidence));

    for (final detection in sorted) {
      _Track? best;
      var bestIou = matchIou;
      for (final track in unmatched) {
        if (track.classIndex != detection.classIndex) continue;
        final iou = _iou(track.box, detection.box);
        if (iou >= bestIou) {
          best = track;
          bestIou = iou;
        }
      }

      if (best != null) {
        if (detection.confidence < keepConfidence) continue;
        unmatched.remove(best);
        best
          ..box = ui.Rect.lerp(best.box, detection.box, smoothing)!
          ..confidence =
              best.confidence +
              (detection.confidence - best.confidence) * smoothing
          ..hits += 1
          ..misses = 0;
      } else if (detection.confidence >= showConfidence) {
        _tracks.add(
          _Track(detection.classIndex, detection.box, detection.confidence),
        );
      }
    }

    for (final track in unmatched) {
      track.misses += 1;
    }
    _tracks.removeWhere((track) => track.misses > maxMisses);

    return [
      for (final track in _tracks)
        if (track.hits >= minHits)
          Detection(
            classIndex: track.classIndex,
            confidence: track.confidence,
            box: track.box,
          ),
    ];
  }

  void clear() => _tracks.clear();

  static double _iou(ui.Rect a, ui.Rect b) {
    final intersection = a.intersect(b);
    if (intersection.width <= 0 || intersection.height <= 0) return 0;
    final interArea = intersection.width * intersection.height;
    final union = a.width * a.height + b.width * b.height - interArea;
    return union <= 0 ? 0 : interArea / union;
  }
}

class _Track {
  _Track(this.classIndex, this.box, this.confidence);

  final int classIndex;
  ui.Rect box;
  double confidence;
  int hits = 1;
  int misses = 0;
}
