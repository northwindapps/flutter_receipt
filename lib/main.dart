import 'dart:io';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:camera/camera.dart';

import 'detector.dart';
import 'tracker.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  // This widget is the root of your application.
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Flutter Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
      ),
      home: const MyHomePage(title: 'Flutter Demo Home Page'),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key, required this.title});

  // This widget is the home page of your application. It is stateful, meaning
  // that it has a State object (defined below) that contains fields that affect
  // how it looks.

  // This class is the configuration for the state. It holds the values (in this
  // case the title) provided by the parent (in this case the App widget) and
  // used by the build method of the State. Fields in a Widget subclass are
  // always marked "final".

  final String title;

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late List<CameraDescription> _cameras;
  CameraController? _controller;
  bool _isCameraInitialized = false;

  // The model is loaded once, in the background, while the camera starts.
  final Future<Detector> _detector = Detector.load();
  Detector? _loadedDetector;
  bool _isDetecting = false;
  DetectionResult? _result;

  // Live detection on the camera stream. Only one frame is processed at a
  // time; frames arriving meanwhile are dropped.
  bool _live = true;
  bool _frameInFlight = false;
  FrameDetections? _liveDetections;
  final DetectionTracker _tracker = DetectionTracker();
  final Stopwatch _frameClock = Stopwatch();
  double _liveFps = 0;

  @override
  void initState() {
    super.initState();
    _detector.then((detector) => _loadedDetector = detector);
    _initCamera();
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(
      _cameras[0],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup:
          Platform.isIOS ? ImageFormatGroup.bgra8888 : ImageFormatGroup.yuv420,
    );

    await _controller!.initialize();

    if (!mounted) return;
    setState(() {
      _isCameraInitialized = true;
    });
    if (_live) await _startLive();
  }

  Future<void> _startLive() async {
    final controller = _controller;
    if (controller == null || controller.value.isStreamingImages) return;
    _frameClock
      ..reset()
      ..start();
    await controller.startImageStream(_onFrame);
  }

  Future<void> _stopLive() async {
    final controller = _controller;
    if (controller != null && controller.value.isStreamingImages) {
      await controller.stopImageStream();
    }
  }

  Future<void> _setLive(bool live) async {
    setState(() {
      _live = live;
      _liveDetections = null;
    });
    _tracker.clear();
    live ? await _startLive() : await _stopLive();
  }

  void _onFrame(CameraImage frame) {
    final detector = _loadedDetector;
    if (detector == null || _frameInFlight || _isDetecting) return;

    _frameInFlight = true;
    detector
        .detectFrame(frame, _frameRotation())
        .then((result) {
          if (!mounted || !_live) return;
          final elapsed = _frameClock.elapsedMilliseconds;
          _frameClock.reset();
          // Boxes from a differently rotated frame can't be matched.
          if (result.frameSize != _liveDetections?.frameSize) _tracker.clear();
          setState(() {
            _liveDetections = FrameDetections(
              result.frameSize,
              _tracker.update(result.detections),
            );
            _liveFps = elapsed > 0 ? 1000 / elapsed : 0;
          });
        }, onError: (Object e) => print('Live detection error: $e'))
        .whenComplete(() => _frameInFlight = false);
  }

  /// Clockwise rotation that makes a stream frame match the screen.
  int _frameRotation() {
    final controller = _controller!;
    const deviceDegrees = {
      DeviceOrientation.portraitUp: 0,
      DeviceOrientation.landscapeLeft: 90,
      DeviceOrientation.portraitDown: 180,
      DeviceOrientation.landscapeRight: 270,
    };
    final device = deviceDegrees[controller.value.deviceOrientation]!;
    final sensor = controller.description.sensorOrientation;
    return controller.description.lensDirection == CameraLensDirection.front
        ? (sensor + device) % 360
        : (sensor - device + 360) % 360;
  }

  @override
  void dispose() {
    _controller?.dispose();
    _detector.then((detector) => detector.dispose());
    _result?.image.dispose();
    super.dispose();
  }

  Future<void> _takePicture() async {
    if (_isDetecting || !_controller!.value.isInitialized) return;

    setState(() => _isDetecting = true);
    try {
      // Pause the frame stream so the capture gets the camera to itself.
      await _stopLive();
      final XFile picture = await _controller!.takePicture();
      print('Picture saved to ${picture.path}');

      final result = await (await _detector).detect(picture.path);
      print('Results: ${result.detections}');

      if (!mounted) {
        result.image.dispose();
        return;
      }
      setState(() {
        _result?.image.dispose();
        _result = result;
      });
    } catch (e) {
      print('Error taking picture: $e');
    } finally {
      if (mounted) {
        setState(() => _isDetecting = false);
        if (_live) await _startLive();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final result = _result;
    return Scaffold(
      appBar: AppBar(
        title: Text(widget.title),
        actions: [
          const Text('Live'),
          Switch(value: _live, onChanged: _setLive),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child:
                _isCameraInitialized
                    ? CameraPreview(
                      _controller!,
                      child:
                          _live && _liveDetections != null
                              ? CustomPaint(
                                painter: _BoxesPainter(
                                  _liveDetections!,
                                  fps: _liveFps,
                                  mirror:
                                      _controller!.description.lensDirection ==
                                      CameraLensDirection.front,
                                ),
                              )
                              : null,
                    )
                    : const Center(child: CircularProgressIndicator()),
          ),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: ElevatedButton.icon(
              onPressed: _isDetecting ? null : _takePicture,
              icon:
                  _isDetecting
                      ? const SizedBox.square(
                        dimension: 18,
                        child: CircularProgressIndicator(strokeWidth: 2),
                      )
                      : const Icon(Icons.camera),
              label: Text(_isDetecting ? 'Detecting…' : 'Take Picture'),
            ),
          ),
          // Cropped detections
          result != null && result.detections.isNotEmpty
              ? SingleChildScrollView(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.all(10),
                child: Row(
                  children:
                      result.detections.map((detection) {
                        return Container(
                          margin: const EdgeInsets.only(right: 10),
                          decoration: BoxDecoration(
                            border: Border.all(color: Colors.grey),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Column(
                            children: [
                              ClipRRect(
                                borderRadius: const BorderRadius.vertical(
                                  top: Radius.circular(10),
                                ),
                                child: CustomPaint(
                                  size: const Size.square(150),
                                  painter: _CropPainter(
                                    result.image,
                                    detection.box,
                                  ),
                                ),
                              ),
                              Text(
                                '${detection.className} '
                                '${(detection.confidence * 100).round()}%',
                              ),
                            ],
                          ),
                        );
                      }).toList(),
                ),
              )
              : const SizedBox.shrink(),
        ],
      ),
    );
  }
}

/// Paints one region of the photo, scaled to cover the canvas.
class _CropPainter extends CustomPainter {
  _CropPainter(this.image, this.region);

  final ui.Image image;
  final Rect region;

  @override
  void paint(Canvas canvas, Size size) {
    final fitted = applyBoxFit(BoxFit.cover, region.size, size);
    final source = Alignment.center.inscribe(fitted.source, region);
    final destination = Alignment.center.inscribe(
      fitted.destination,
      Offset.zero & size,
    );
    canvas.drawImageRect(
      image,
      source,
      destination,
      Paint()..filterQuality = FilterQuality.medium,
    );
  }

  @override
  bool shouldRepaint(_CropPainter oldDelegate) =>
      image != oldDelegate.image || region != oldDelegate.region;
}

/// Draws live detections over the camera preview.
class _BoxesPainter extends CustomPainter {
  _BoxesPainter(this.result, {required this.fps, required this.mirror});

  final FrameDetections result;
  final double fps;
  final bool mirror;

  @override
  void paint(Canvas canvas, Size size) {
    final scaleX = size.width / result.frameSize.width;
    final scaleY = size.height / result.frameSize.height;

    for (final detection in result.detections) {
      var rect = Rect.fromLTRB(
        detection.box.left * scaleX,
        detection.box.top * scaleY,
        detection.box.right * scaleX,
        detection.box.bottom * scaleY,
      );
      if (mirror) {
        rect = Rect.fromLTRB(
          size.width - rect.right,
          rect.top,
          size.width - rect.left,
          rect.bottom,
        );
      }
      final color =
          Colors.primaries[detection.classIndex * 3 % Colors.primaries.length];
      canvas.drawRect(
        rect,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = 2,
      );
      _drawLabel(
        canvas,
        '${detection.className} ${(detection.confidence * 100).round()}%',
        rect.topLeft,
        color,
      );
    }
    _drawLabel(
      canvas,
      '${fps.toStringAsFixed(1)} fps',
      const Offset(8, 8),
      Colors.black54,
    );
  }

  void _drawLabel(Canvas canvas, String text, Offset at, Color background) {
    final painter = TextPainter(
      text: TextSpan(
        text: text,
        style: const TextStyle(color: Colors.white, fontSize: 12),
      ),
      textDirection: TextDirection.ltr,
    )..layout();
    final origin = Offset(at.dx, (at.dy - painter.height).clamp(0, at.dy));
    canvas.drawRect(origin & painter.size, Paint()..color = background);
    painter.paint(canvas, origin);
    painter.dispose();
  }

  @override
  bool shouldRepaint(_BoxesPainter oldDelegate) =>
      result != oldDelegate.result || fps != oldDelegate.fps;
}
