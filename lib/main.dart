import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:io';
import 'package:image/image.dart' as img;

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
  XFile? _capturedImage;

  Interpreter? _interpreter;
  List<String> labels = [
    'class0',
    'class1',
    'class2',
    'class3',
    'class4',
    'class5',
    'class6',
    'class7',
    'class8',
    'class9',
    // Change this labels as you need
  ];

  // Model configurations
  final int inputSize =
      416; // Common YOLO input size, adjust based on your model
  final double confidenceThreshold = 0.5;

  Uint8List? _croppedImage;

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
    super.dispose();
  }

  Future<void> _takePicture() async {
    if (!_controller!.value.isInitialized) return;

    try {
      final XFile picture = await _controller!.takePicture();
      setState(() {
        _capturedImage = picture;
      });

      // inference starts
      _interpreter = await Interpreter.fromAsset(
        'yolo11n_saved_model/best_float32.tflite',
      );

      print('Picture saved to ${picture.path}');

      final File imageFile = File(picture.path);

      final detections = await runDetection(imageFile);
    } catch (e) {
      print('Error taking picture: $e');
    }
  }

  Future<List<dynamic>> runDetection(File imageFile) async {
    if (_interpreter == null) {
      throw Exception("Interpreter not initialized");
    }

    try {
      // Get model input details to determine the expected shape
      final inputShape = _interpreter!.getInputTensor(0).shape;
      print('Model expects input shape: $inputShape');

      // Calculate input dimensions from the shape
      // Typically [1, height, width, channels] or [batch, height, width, channels]
      final inputHeight = inputShape[1];
      final inputWidth = inputShape[2];

      print('Using input dimensions: $inputWidth x $inputHeight');

      // Decode the image
      final imageData = await imageFile.readAsBytes();
      final image = img.decodeImage(imageData);
      if (image == null) {
        throw Exception("Failed to decode image");
      }

      // Resize to the exact dimensions expected by the model
      final resizedImage = img.copyResize(
        image,
        width: inputWidth,
        height: inputHeight,
        interpolation: img.Interpolation.linear,
      );

      // Convert image to input tensor format
      final input = imageToByteList(resizedImage, inputWidth, inputHeight);

      // Get output shape from the model
      final outputTensors = _interpreter!.getOutputTensors();
      final outputShape = outputTensors[0].shape;
      print('Model output shape: $outputShape');

      // Create output buffer with the correct shape
      final output = List.filled(
        outputShape.reduce((a, b) => a * b),
        0.0,
      ).reshape(outputShape);

      // Run inference
      _interpreter!.run(input, output);

      // Process the output to get detection boxes
      final results = processOutput(output, image.width, image.height);
      if (results != []) {
        print('Results: $results');
        print('width' + image.width.toString());
        print('height' + image.height.toString());
        final box = List<int>.from(results[0]['box']);
        // If the box is relative to the original image, make sure it’s also resized
        final int x = box[0];
        final int y = box[1];
        final int width = box[2];
        final int height = box[3];

        // Optionally clamp (resizedImage has inputWidth x inputHeight dimensions)
        final cropped = img.copyCrop(
          image,
          x: (x - (width / 2)).toInt(),
          y: (y - (height / 2)).toInt(),
          width: width,
          height: height,
        );

        //Convert to JPG
        final jpg = img.encodeJpg(cropped);
        setState(() {
          _croppedImage = Uint8List.fromList(jpg);
        });
      }
      return results;
    } catch (e, stackTrace) {
      print('Error in runDetection: $e');
      print('Stack trace: $stackTrace');
      rethrow;
    }
  }

  // Updated to take dimensions as parameters
  List<dynamic> imageToByteList(img.Image image, int width, int height) {
    // Get bytes with the correct channel order (RGB for most YOLO models)
    final bytes = image.getBytes(order: img.ChannelOrder.rgb);

    // Convert to float32 and normalize to 0-1 range
    final inputChannels = 3; // RGB
    final batchSize = 1;

    final convertedBytes = Float32List(
      batchSize * height * width * inputChannels,
    );

    int pixelIndex = 0;
    for (int i = 0; i < bytes.length; i++) {
      // Normalize from 0-255 to 0-1
      convertedBytes[pixelIndex++] = bytes[i] / 255.0;
    }

    // Reshape to the format expected by TFLite
    return convertedBytes.reshape([batchSize, height, width, inputChannels]);
  }

  List<dynamic> processOutput(
    List<dynamic> output,
    int imageWidth,
    int imageHeight,
  ) {
    // This is a placeholder - actual processing depends on YOLO version
    // You'll need to implement non-max suppression and bounding box decoding

    // List<dynamic> detections = [];
    // Process output to get bounding boxes, classes, and confidence scores
    // ...

    try {
      // Try both YOLOv8 output interpretations
      print('Trying standard YOLOv8 interpretation (classes in rows 4-83)');
      final standardDetections = processYoloV8Standard(
        output,
        imageWidth,
        imageHeight,
      );
      return standardDetections;
    } catch (e, stackTrace) {
      print('Error processing output: $e');
      print('Stack trace: $stackTrace');
      return [];
    }
  }

  List<dynamic> processYoloV8Standard(
    List<dynamic> output,
    int imageWidth,
    int imageHeight,
  ) {
    List<dynamic> detections = [];

    try {
      // YOLOv8 output format is [1, 84, 8400]
      // Where 84 = 4 (box coordinates) + 80 (class probabilities)
      final numClasses = output[0].length - 4;
      final numDetections = output[0][0].length;

      print(
        'Processing YOLOv8 format: $numDetections detections, $numClasses classes',
      );

      // Process each potential detection
      for (int i = 0; i < numDetections; i++) {
        // Get bounding box coordinates
        final x = output[0][0][i]; // center x
        final y = output[0][1][i]; // center y
        final w = output[0][2][i]; // width
        final h = output[0][3][i]; // height

        // Skip if box dimensions are too small
        if (w < 0.01 || h < 0.01) continue;

        // Find the class with highest probability
        double maxClassProb = 0;
        int classIndex = 0;

        for (int c = 0; c < numClasses; c++) {
          final classProb = output[0][c + 4][i];
          if (classProb > maxClassProb) {
            maxClassProb = classProb;
            classIndex = c;
          }
        }

        // If class probability is good enough
        if (maxClassProb >= confidenceThreshold) {
          final centerX = (x * imageWidth).round();
          final centerY = (y * imageHeight).round();
          final width = (w * imageWidth).round();
          final height = (h * imageHeight).round();

          // Only add if the box has positive area
          if (centerX > 0 && centerY > 0 && width > 0 && height > 0) {
            detections.add({
              'index': classIndex,
              'className':
                  classIndex < labels.length ? labels[classIndex] : 'Unknown',
              'confidence': maxClassProb,
              'box': [centerX, centerY, width, height],
            });
          }
        }
      }

      print('Found ${detections.length} detections before NMS');

      // Apply non-maximum suppression to remove overlapping boxes
      print('box' + detections.toString());
      final result = _nonMaxSuppression(detections, 0.5);
      print('Found ${result.length} detections after NMS');

      return result;
    } catch (e) {
      print('Error in standard processing: $e');
      return [];
    }
  }

  List<dynamic> _nonMaxSuppression(List<dynamic> boxes, double threshold) {
    // Sort by confidence
    boxes.sort((a, b) => b['confidence'].compareTo(a['confidence']));

    List<dynamic> selected = [];
    List<bool> suppressed = List.filled(boxes.length, false);

    for (int i = 0; i < boxes.length; i++) {
      if (suppressed[i]) continue;

      selected.add(boxes[i]);

      for (int j = i + 1; j < boxes.length; j++) {
        if (suppressed[j]) continue;

        // Calculate IoU (Intersection over Union)
        final iou = _calculateIoU(
          List<int>.from(boxes[i]['box']),
          List<int>.from(boxes[j]['box']),
        );
        if (iou >= threshold) {
          suppressed[j] = true;
        }
      }
    }

    return selected;
  }

  double _calculateIoU(List<int> boxA, List<int> boxB) {
    // Calculate intersection area
    final xA = math.max(boxA[0], boxB[0]);
    final yA = math.max(boxA[1], boxB[1]);
    final xB = math.min(boxA[2], boxB[2]);
    final yB = math.min(boxA[3], boxB[3]);

    final interArea = math.max(0, xB - xA) * math.max(0, yB - yA);

    // Calculate union area
    final boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    final boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);

    final unionArea = boxAArea + boxBArea - interArea;

    return interArea / unionArea;
  }

  @override
  Widget build(BuildContext context) {
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
          if (_capturedImage != null)
            Image.file(File(_capturedImage!.path), height: 150),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: ElevatedButton.icon(
              onPressed: _takePicture,
              icon: const Icon(Icons.camera),
              label: const Text('Take Picture'),
            ),
          ),

          // Captured and Cropped Image Display
          if (_croppedImage != null)
            Container(
              height: 300,
              width: double.infinity,
              margin: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                border: Border.all(color: Colors.grey),
                borderRadius: BorderRadius.circular(10),
              ),
              child: Image.memory(
                _croppedImage!, // Uint8List of the cropped image
                fit: BoxFit.contain,
                height: 300,
              ),
            ),
        ],
      ),
    );
  }
}
