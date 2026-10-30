import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

const _modelAsset = 'yolo11n_saved_model/best_float32.tflite';

// Class names from the Ultralytics metadata embedded in the model file.
const detectionLabels = ['datechar', 'datesla', 'item', 'itemv', 'totals', 'tv'];

// Each detection travels between isolates as 6 floats:
// class, confidence, left, top, right, bottom (box normalized to 0-1).
const _fieldsPerDetection = 6;

class Detection {
  const Detection({
    required this.classIndex,
    required this.confidence,
    required this.box,
  });

  final int classIndex;
  final double confidence;

  /// Bounding box in pixels of the original photo.
  final ui.Rect box;

  String get className =>
      classIndex < detectionLabels.length
          ? detectionLabels[classIndex]
          : 'Unknown';

  @override
  String toString() =>
      '$className ${confidence.toStringAsFixed(2)} '
      '[${box.left.round()}, ${box.top.round()}, '
      '${box.right.round()}, ${box.bottom.round()}]';
}

class DetectionResult {
  DetectionResult(this.image, this.detections);

  /// The decoded photo. The caller owns it and must dispose it.
  final ui.Image image;
  final List<Detection> detections;
}

/// Runs the YOLO model in a background isolate that keeps the interpreter
/// loaded, so the UI thread never blocks on model loading or inference.
class Detector {
  Detector._(this._isolate, this._worker, this._inputWidth, this._inputHeight);

  final Isolate _isolate;
  final SendPort _worker;
  final int _inputWidth;
  final int _inputHeight;

  static Future<Detector> load({
    double confidenceThreshold = 0.5,
    double iouThreshold = 0.5,
  }) async {
    final stopwatch = Stopwatch()..start();
    final model = await rootBundle.load(_modelAsset);
    final modelBytes = model.buffer.asUint8List(
      model.offsetInBytes,
      model.lengthInBytes,
    );

    final ready = ReceivePort();
    final isolate = await Isolate.spawn(_workerMain, (
      ready.sendPort,
      TransferableTypedData.fromList([modelBytes]),
      confidenceThreshold,
      iouThreshold,
    ), debugName: 'yolo-detector');
    final message = await ready.first;
    ready.close();

    if (message is String) {
      isolate.kill();
      throw Exception('Failed to load model: $message');
    }
    final (port, width, height) = message as (SendPort, int, int);
    debugPrint(
      'Detector ready in ${stopwatch.elapsedMilliseconds} ms '
      '(input ${width}x$height)',
    );
    return Detector._(isolate, port, width, height);
  }

  Future<DetectionResult> detect(String imagePath) async {
    final total = Stopwatch()..start();
    final bytes = await File(imagePath).readAsBytes();

    // Flutter's native decoders do the JPEG decode and the resize to the
    // model's input size, instead of the much slower pure-Dart decoder.
    final (image, modelInput) =
        await (
          _decode(bytes),
          _decode(bytes, width: _inputWidth, height: _inputHeight),
        ).wait;
    final rgba = await modelInput.toByteData(
      format: ui.ImageByteFormat.rawRgba,
    );
    modelInput.dispose();
    final decodeMs = total.elapsedMilliseconds;

    final reply = ReceivePort();
    _worker.send((
      TransferableTypedData.fromList([
        rgba!.buffer.asUint8List(rgba.offsetInBytes, rgba.lengthInBytes),
      ]),
      reply.sendPort,
    ));
    final message = await reply.first;
    reply.close();

    if (message is String) {
      image.dispose();
      throw Exception('Detection failed: $message');
    }
    final (boxes, preprocessMs, inferenceMs, postprocessMs) =
        message as (Float32List, int, int, int);

    final detections = <Detection>[];
    for (var i = 0; i < boxes.length; i += _fieldsPerDetection) {
      final box = ui.Rect.fromLTRB(
        boxes[i + 2] * image.width,
        boxes[i + 3] * image.height,
        boxes[i + 4] * image.width,
        boxes[i + 5] * image.height,
      ).intersect(
        ui.Rect.fromLTWH(0, 0, image.width.toDouble(), image.height.toDouble()),
      );
      if (box.width < 1 || box.height < 1) continue;
      detections.add(
        Detection(
          classIndex: boxes[i].toInt(),
          confidence: boxes[i + 1],
          box: box,
        ),
      );
    }

    debugPrint(
      'Detection: decode $decodeMs ms, preprocess $preprocessMs ms, '
      'inference $inferenceMs ms, postprocess $postprocessMs ms, '
      'total ${total.elapsedMilliseconds} ms, '
      '${detections.length} found ${image.width}x${image.height}',
    );
    return DetectionResult(image, detections);
  }

  void dispose() {
    _worker.send(null);
    _isolate.kill();
  }

  static Future<ui.Image> _decode(
    Uint8List bytes, {
    int? width,
    int? height,
  }) async {
    final codec = await ui.instantiateImageCodec(
      bytes,
      targetWidth: width,
      targetHeight: height,
    );
    final frame = await codec.getNextFrame();
    codec.dispose();
    return frame.image;
  }
}

void _workerMain((SendPort, TransferableTypedData, double, double) args) {
  final (ready, modelData, confidenceThreshold, iouThreshold) = args;

  final Interpreter interpreter;
  try {
    interpreter = _createInterpreter(modelData.materialize().asUint8List());
    if (interpreter.getInputTensor(0).type != TensorType.float32) {
      throw Exception('Expected a float32 model input');
    }
  } catch (e) {
    ready.send('$e');
    return;
  }

  final inputShape = interpreter.getInputTensor(0).shape; // [1, h, w, 3]
  final outputShape = interpreter.getOutputTensor(0).shape; // [1, 4 + classes, anchors]
  final input = Float32List(inputShape[1] * inputShape[2] * 3);

  final requests = ReceivePort();
  ready.send((requests.sendPort, inputShape[2], inputShape[1]));

  requests.listen((message) {
    if (message == null) {
      interpreter.close();
      requests.close();
      return;
    }
    final (data, reply) = message as (TransferableTypedData, SendPort);
    try {
      final stopwatch = Stopwatch()..start();
      final rgba = data.materialize().asUint8List();
      for (var p = 0, i = 0; p < rgba.length; p += 4) {
        input[i++] = rgba[p] / 255.0;
        input[i++] = rgba[p + 1] / 255.0;
        input[i++] = rgba[p + 2] / 255.0;
      }
      // Write straight into the tensor's memory instead of nested lists.
      interpreter.getInputTensor(0).data = input.buffer.asUint8List();
      final preprocessMs = stopwatch.elapsedMilliseconds;

      stopwatch.reset();
      interpreter.invoke();
      final inferenceMs = stopwatch.elapsedMilliseconds;

      stopwatch.reset();
      final outputBytes = interpreter.getOutputTensor(0).data;
      final output = outputBytes.buffer.asFloat32List(
        outputBytes.offsetInBytes,
        outputBytes.lengthInBytes ~/ Float32List.bytesPerElement,
      );
      final boxes = _decodeYolo(
        output,
        outputShape[1],
        outputShape[2],
        confidenceThreshold,
        iouThreshold,
      );
      reply.send((
        boxes,
        preprocessMs,
        inferenceMs,
        stopwatch.elapsedMilliseconds,
      ));
    } catch (e, stackTrace) {
      reply.send('$e\n$stackTrace');
    }
  });
}

Interpreter _createInterpreter(Uint8List model) {
  final threads = math.min(4, Platform.numberOfProcessors);
  if (Platform.isAndroid || Platform.isIOS) {
    try {
      final options =
          InterpreterOptions()
            ..threads = threads
            ..addDelegate(
              XNNPackDelegate(
                options: XNNPackDelegateOptions(numThreads: threads),
              ),
            );
      return Interpreter.fromBuffer(model, options: options);
    } catch (e) {
      debugPrint('XNNPACK unavailable, using the default CPU kernels: $e');
    }
  }
  return Interpreter.fromBuffer(
    model,
    options: InterpreterOptions()..threads = threads,
  );
}

/// Decodes a YOLOv8/YOLO11 output laid out as [4 + classes][anchors], where the
/// first 4 rows are normalized center x, center y, width and height, then
/// applies per-class non-max suppression.
Float32List _decodeYolo(
  Float32List output,
  int rows,
  int anchors,
  double confidenceThreshold,
  double iouThreshold,
) {
  final numClasses = rows - 4;
  final candidates = <int>[];
  final classes = Int32List(anchors);
  final scores = Float32List(anchors);

  for (var i = 0; i < anchors; i++) {
    final w = output[2 * anchors + i];
    final h = output[3 * anchors + i];
    if (w < 0.01 || h < 0.01) continue;

    var bestScore = 0.0;
    var bestClass = 0;
    for (var c = 0; c < numClasses; c++) {
      final score = output[(c + 4) * anchors + i];
      if (score > bestScore) {
        bestScore = score;
        bestClass = c;
      }
    }
    if (bestScore >= confidenceThreshold) {
      candidates.add(i);
      classes[i] = bestClass;
      scores[i] = bestScore;
    }
  }
  candidates.sort((a, b) => scores[b].compareTo(scores[a]));

  ui.Rect boxAt(int i) {
    final cx = output[i];
    final cy = output[anchors + i];
    final w = output[2 * anchors + i];
    final h = output[3 * anchors + i];
    return ui.Rect.fromLTWH(cx - w / 2, cy - h / 2, w, h);
  }

  final kept = <int>[];
  final keptBoxes = <ui.Rect>[];
  for (final i in candidates) {
    final box = boxAt(i);
    var suppressed = false;
    for (var k = 0; k < kept.length; k++) {
      if (classes[kept[k]] == classes[i] &&
          _iou(box, keptBoxes[k]) >= iouThreshold) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) {
      kept.add(i);
      keptBoxes.add(box);
    }
  }

  final result = Float32List(kept.length * _fieldsPerDetection);
  for (var k = 0; k < kept.length; k++) {
    final box = keptBoxes[k];
    result.setAll(k * _fieldsPerDetection, [
      classes[kept[k]].toDouble(),
      scores[kept[k]],
      box.left,
      box.top,
      box.right,
      box.bottom,
    ]);
  }
  return result;
}

double _iou(ui.Rect a, ui.Rect b) {
  final intersection = a.intersect(b);
  if (intersection.width <= 0 || intersection.height <= 0) return 0;
  final interArea = intersection.width * intersection.height;
  final union = a.width * a.height + b.width * b.height - interArea;
  return union <= 0 ? 0 : interArea / union;
}
