import 'dart:io';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

const _modelAsset = 'models/cord_yolo11n.tflite';

// CORD-v2 categories, in the class order of training/cord_to_yolo.py.
const detectionLabels = [
  'menu.nm',
  'menu.cnt',
  'menu.price',
  'menu.unitprice',
  'menu.discountprice',
  'menu.num',
  'menu.sub.nm',
  'menu.sub.cnt',
  'menu.sub.price',
  'sub_total.subtotal_price',
  'sub_total.tax_price',
  'sub_total.service_price',
  'total.total_price',
  'total.cashprice',
  'total.changeprice',
  'total.creditcardprice',
  'total.menuqty_cnt',
];

// The worker keeps boxes down to this confidence, so live tracking can keep
// showing a box whose score dips for a frame. Callers apply their own cut-off.
const minWorkerConfidence = 0.25;

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

class FrameDetections {
  FrameDetections(this.frameSize, this.detections);

  /// Size of the upright (rotated) camera frame the boxes refer to.
  final ui.Size frameSize;
  final List<Detection> detections;
}

/// Runs the YOLO model in a background isolate that keeps the interpreter
/// loaded, so the UI thread never blocks on model loading or inference.
class Detector {
  Detector._(this._isolate, this._worker, this.confidenceThreshold);

  final Isolate _isolate;
  final SendPort _worker;

  /// Minimum confidence for detections returned by [detect].
  final double confidenceThreshold;

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
    return Detector._(isolate, port, confidenceThreshold);
  }

  Future<DetectionResult> detect(String imagePath) async {
    final total = Stopwatch()..start();
    final bytes = await File(imagePath).readAsBytes();

    // Flutter's native decoder is much faster than the pure-Dart one.
    final image = await _decode(bytes);
    final rgba = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    final decodeMs = total.elapsedMilliseconds;

    final _WorkerResult result;
    try {
      result = await _run(
        _Request(_Format.rgba, [
          TransferableTypedData.fromList([
            rgba!.buffer.asUint8List(rgba.offsetInBytes, rgba.lengthInBytes),
          ]),
        ], image.width, image.height, rowStride: image.width * 4),
      );
    } catch (_) {
      image.dispose();
      rethrow;
    }
    final (boxes, preprocessMs, inferenceMs, postprocessMs) = result;
    final detections =
        _toDetections(
          boxes,
          ui.Size(image.width.toDouble(), image.height.toDouble()),
        ).where((d) => d.confidence >= confidenceThreshold).toList();

    debugPrint(
      'Detection: decode $decodeMs ms, preprocess $preprocessMs ms, '
      'inference $inferenceMs ms, postprocess $postprocessMs ms, '
      'total ${total.elapsedMilliseconds} ms, '
      '${detections.length} found ${image.width}x${image.height}',
    );
    return DetectionResult(image, detections);
  }

  /// Detects objects in a camera stream frame. [rotation] is the clockwise
  /// rotation in degrees that makes the frame upright. Returns detections down
  /// to [minWorkerConfidence]; filter or track them before showing.
  ///
  /// The frame's bytes are copied before this method first awaits, so the
  /// caller may let the camera reuse the buffer right after calling it.
  Future<FrameDetections> detectFrame(CameraImage frame, int rotation) async {
    final _Request request;
    switch (frame.format.group) {
      case ImageFormatGroup.yuv420 when frame.planes.length == 3:
        request = _Request(
          _Format.yuv420,
          [
            for (final plane in frame.planes)
              TransferableTypedData.fromList([plane.bytes]),
          ],
          frame.width,
          frame.height,
          rowStride: frame.planes[0].bytesPerRow,
          uvRowStride: frame.planes[1].bytesPerRow,
          uvPixelStride: frame.planes[1].bytesPerPixel ?? 1,
          rotation: rotation,
        );
      case ImageFormatGroup.bgra8888:
        request = _Request(
          _Format.bgra8888,
          [TransferableTypedData.fromList([frame.planes[0].bytes])],
          frame.width,
          frame.height,
          rowStride: frame.planes[0].bytesPerRow,
          rotation: rotation,
        );
      default:
        throw UnsupportedError(
          'Unsupported camera frame format ${frame.format.group}',
        );
    }

    final (boxes, _, _, _) = await _run(request);
    final upright =
        rotation % 180 == 0
            ? ui.Size(frame.width.toDouble(), frame.height.toDouble())
            : ui.Size(frame.height.toDouble(), frame.width.toDouble());
    return FrameDetections(upright, _toDetections(boxes, upright));
  }

  Future<_WorkerResult> _run(_Request request) async {
    final reply = ReceivePort();
    _worker.send((request, reply.sendPort));
    final message = await reply.first;
    reply.close();
    if (message is String) throw Exception('Detection failed: $message');
    return message as _WorkerResult;
  }

  static List<Detection> _toDetections(Float32List boxes, ui.Size size) {
    final detections = <Detection>[];
    for (var i = 0; i < boxes.length; i += _fieldsPerDetection) {
      final box = ui.Rect.fromLTRB(
        boxes[i + 2] * size.width,
        boxes[i + 3] * size.height,
        boxes[i + 4] * size.width,
        boxes[i + 5] * size.height,
      ).intersect(ui.Offset.zero & size);
      if (box.width < 1 || box.height < 1) continue;
      detections.add(
        Detection(
          classIndex: boxes[i].toInt(),
          confidence: boxes[i + 1],
          box: box,
        ),
      );
    }
    return detections;
  }

  void dispose() {
    _worker.send(null);
    _isolate.kill();
  }

  static Future<ui.Image> _decode(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    codec.dispose();
    return frame.image;
  }
}

enum _Format { rgba, yuv420, bgra8888 }

/// Pixels sent to the worker, which converts, rotates and letterboxes them
/// into the model input.
class _Request {
  _Request(
    this.format,
    this.planes,
    this.width,
    this.height, {
    this.rowStride = 0,
    this.uvRowStride = 0,
    this.uvPixelStride = 0,
    this.rotation = 0,
  });

  final _Format format;
  final List<TransferableTypedData> planes;
  final int width;
  final int height;
  final int rowStride;
  final int uvRowStride;
  final int uvPixelStride;
  final int rotation;
}

/// Detections, then preprocess, inference and postprocess times in ms.
typedef _WorkerResult = (Float32List, int, int, int);

void _workerMain((SendPort, TransferableTypedData, double) args) {
  final (ready, modelData, iouThreshold) = args;

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

  // [1, h, w, 3] from the TFLite exporter, [1, 3, h, w] from the LiteRT one.
  final inputShape = interpreter.getInputTensor(0).shape;
  final channelsFirst = inputShape[1] == 3 && inputShape[3] != 3;
  final inputHeight = channelsFirst ? inputShape[2] : inputShape[1];
  final inputWidth = channelsFirst ? inputShape[3] : inputShape[2];
  final outputShape = interpreter.getOutputTensor(0).shape; // [1, 4 + classes, anchors]
  final input = Float32List(inputWidth * inputHeight * 3);

  final requests = ReceivePort();
  ready.send((requests.sendPort, inputWidth, inputHeight));

  requests.listen((message) {
    if (message == null) {
      interpreter.close();
      requests.close();
      return;
    }
    final (request, reply) = message as (_Request, SendPort);
    try {
      final stopwatch = Stopwatch()..start();
      final letterbox = _fillInput(
        request,
        input,
        inputWidth,
        inputHeight,
        channelsFirst,
      );
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
        inputWidth,
        inputHeight,
        minWorkerConfidence,
        iouThreshold,
      );
      letterbox.toFrame(boxes);
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

/// Where the upright image landed inside the model input.
class _Letterbox {
  _Letterbox(
    this.scale,
    this.padX,
    this.padY,
    this.inputWidth,
    this.inputHeight,
    this.imageWidth,
    this.imageHeight,
  );

  final double scale;
  final int padX, padY, inputWidth, inputHeight, imageWidth, imageHeight;

  /// Converts box coordinates in [boxes] from normalized model-input space to
  /// normalized image space, in place.
  void toFrame(Float32List boxes) {
    for (var i = 0; i < boxes.length; i += _fieldsPerDetection) {
      boxes[i + 2] = (boxes[i + 2] * inputWidth - padX) / scale / imageWidth;
      boxes[i + 3] = (boxes[i + 3] * inputHeight - padY) / scale / imageHeight;
      boxes[i + 4] = (boxes[i + 4] * inputWidth - padX) / scale / imageWidth;
      boxes[i + 5] = (boxes[i + 5] * inputHeight - padY) / scale / imageHeight;
    }
  }
}

/// Writes [request]'s pixels into [input] as normalized RGB floats of size
/// [outWidth] x [outHeight]: rotated upright, scaled to fit while keeping the
/// aspect ratio, and padded with gray, like Ultralytics' letterbox.
/// [channelsFirst] writes planar RRR..GGG..BBB instead of interleaved RGB.
_Letterbox _fillInput(
  _Request request,
  Float32List input,
  int outWidth,
  int outHeight,
  bool channelsFirst,
) {
  final planes = [for (final p in request.planes) p.materialize().asUint8List()];
  final w = request.width;
  final h = request.height;
  final rotation = request.rotation % 360;
  final uprightWidth = rotation % 180 == 0 ? w : h;
  final uprightHeight = rotation % 180 == 0 ? h : w;

  final scale = math.min(outWidth / uprightWidth, outHeight / uprightHeight);
  final scaledWidth = math.min(outWidth, (uprightWidth * scale).round());
  final scaledHeight = math.min(outHeight, (uprightHeight * scale).round());
  final padX = (outWidth - scaledWidth) ~/ 2;
  final padY = (outHeight - scaledHeight) ~/ 2;
  input.fillRange(0, input.length, 114 / 255.0);

  // Offsets between a pixel's R, G and B values, and from one pixel to the next.
  final planeSize = outWidth * outHeight;
  final channelStep = channelsFirst ? planeSize : 1;
  final pixelStep = channelsFirst ? 1 : 3;

  for (var oy = 0; oy < scaledHeight; oy++) {
    final uy = math.min(uprightHeight - 1, (oy / scale).floor());
    var i = ((oy + padY) * outWidth + padX) * pixelStep;
    for (var ox = 0; ox < scaledWidth; ox++) {
      final ux = math.min(uprightWidth - 1, (ox / scale).floor());
      // Map the upright pixel back to the source's orientation.
      final int sx, sy;
      switch (rotation) {
        case 90:
          sx = uy;
          sy = h - 1 - ux;
        case 180:
          sx = w - 1 - ux;
          sy = h - 1 - uy;
        case 270:
          sx = w - 1 - uy;
          sy = ux;
        default:
          sx = ux;
          sy = uy;
      }

      double r, g, b;
      switch (request.format) {
        case _Format.yuv420:
          final y = planes[0][sy * request.rowStride + sx];
          final uvIndex =
              (sy >> 1) * request.uvRowStride +
              (sx >> 1) * request.uvPixelStride;
          final u = planes[1][uvIndex] - 128;
          final v = planes[2][uvIndex] - 128;
          r = y + 1.402 * v;
          g = y - 0.344136 * u - 0.714136 * v;
          b = y + 1.772 * u;
        case _Format.bgra8888:
          final p = sy * request.rowStride + sx * 4;
          b = planes[0][p].toDouble();
          g = planes[0][p + 1].toDouble();
          r = planes[0][p + 2].toDouble();
        case _Format.rgba:
          final p = sy * request.rowStride + sx * 4;
          r = planes[0][p].toDouble();
          g = planes[0][p + 1].toDouble();
          b = planes[0][p + 2].toDouble();
      }
      input[i] = r.clamp(0, 255) / 255.0;
      input[i + channelStep] = g.clamp(0, 255) / 255.0;
      input[i + 2 * channelStep] = b.clamp(0, 255) / 255.0;
      i += pixelStep;
    }
  }
  return _Letterbox(
    scale,
    padX,
    padY,
    outWidth,
    outHeight,
    uprightWidth,
    uprightHeight,
  );
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
/// first 4 rows are center x, center y, width and height, then applies
/// per-class non-max suppression. Returned boxes are normalized to 0-1.
Float32List _decodeYolo(
  Float32List output,
  int rows,
  int anchors,
  int inputWidth,
  int inputHeight,
  double confidenceThreshold,
  double iouThreshold,
) {
  final numClasses = rows - 4;

  // The TFLite exporter emits normalized boxes; other exporters may emit
  // input pixels. Centers beyond 1.5 can only be pixels, so scale those down.
  var maxCenter = 0.0;
  for (var i = 0; i < 2 * anchors; i++) {
    if (output[i] > maxCenter) maxCenter = output[i];
  }
  // (The output is a read-only view of the tensor, so scale on read.)
  final xScale = maxCenter > 1.5 ? 1 / inputWidth : 1.0;
  final yScale = maxCenter > 1.5 ? 1 / inputHeight : 1.0;

  final candidates = <int>[];
  final classes = Int32List(anchors);
  final scores = Float32List(anchors);

  for (var i = 0; i < anchors; i++) {
    final w = output[2 * anchors + i] * xScale;
    final h = output[3 * anchors + i] * yScale;
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
    final cx = output[i] * xScale;
    final cy = output[anchors + i] * yScale;
    final w = output[2 * anchors + i] * xScale;
    final h = output[3 * anchors + i] * yScale;
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
