import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

import 'detector.dart';

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
  bool _isDetecting = false;
  DetectionResult? _result;

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    _cameras = await availableCameras();
    _controller = CameraController(_cameras[0], ResolutionPreset.medium);

    await _controller!.initialize();

    if (!mounted) return;
    setState(() {
      _isCameraInitialized = true;
    });
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
      if (mounted) setState(() => _isDetecting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final result = _result;
    return Scaffold(
      appBar: AppBar(title: Text(widget.title)),
      body: Column(
        children: [
          Expanded(
            child:
                _isCameraInitialized
                    ? CameraPreview(_controller!)
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
