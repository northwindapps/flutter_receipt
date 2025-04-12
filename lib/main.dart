import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:io';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart';
import 'package:image/image.dart' as img;

// https://app.outlier.ai/playground/67f9e004307fdd9ef659d0a9
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
        // This is the theme of your application.
        //
        // TRY THIS: Try running your application with "flutter run". You'll see
        // the application has a purple toolbar. Then, without quitting the app,
        // try changing the seedColor in the colorScheme below to Colors.green
        // and then invoke "hot reload" (save your changes or press the "hot
        // reload" button in a Flutter-supported IDE, or press "r" if you used
        // the command line to start the app).
        //
        // Notice that the counter didn't reset back to zero; the application
        // state is not lost during the reload. To reset the state, use hot
        // restart instead.
        //
        // This works for code too, not just values: Most code changes can be
        // tested with just a hot reload.
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
  List<String>? _labels;

  // Model configurations
  final int inputSize =
      416; // Common YOLO input size, adjust based on your model
  final double confidenceThreshold = 0.5;

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

  Future<void> testWithSampleImage() async {
    try {
      // Load a sample image from assets
      final ByteData data = await rootBundle.load(
        'yolo11n_saved_model/assets/test.jpg',
      );
      final List<int> bytes = data.buffer.asUint8List();

      // Create a temporary file
      final tempDir = await getTemporaryDirectory();
      final tempFile = File('${tempDir.path}/test_byte.jpg');
      await tempFile.writeAsBytes(bytes);

      // Run detection
      final detections = await runDetection(tempFile);

      print('Test image detections: $detections');
    } catch (e) {
      print('Error testing with sample image: $e');
    }
  }

  Future<void> _takePicture() async {
    if (!_controller!.value.isInitialized) return;

    final directory = await getTemporaryDirectory();
    final String filePath = join(directory.path, '${DateTime.now()}.png');

    try {
      final XFile picture = await _controller!.takePicture();
      setState(() {
        _capturedImage = picture;
      });

      // inference starts
      final interpreter = await Interpreter.fromAsset(
        'yolo11n_saved_model/yolo11n_float32.tflite',
      );

      _interpreter = interpreter;

      print('Picture saved to ${picture.path}');

      final File imageFile = File(picture.path);

      // final detections = await runDetection(imageFile);

      await testWithSampleImage();
    } catch (e) {
      print('Error taking picture: $e');
    }
  }

  void inspectModelOutput(List<dynamic> output) {
    try {
      print('Detailed output inspection:');

      // Check if output contains any non-zero values
      bool hasNonZeroValues = false;
      double maxValue = 0.0;
      int maxValueRow = -1;
      int maxValueCol = -1;

      for (int i = 0; i < output[0].length; i++) {
        for (int j = 0; j < output[0][i].length; j++) {
          if (output[0][i][j] != 0) {
            hasNonZeroValues = true;
            if (output[0][i][j] > maxValue) {
              maxValue = output[0][i][j];
              maxValueRow = i;
              maxValueCol = j;
            }
          }
        }
      }

      print('  Has non-zero values: $hasNonZeroValues');
      if (hasNonZeroValues) {
        print(
          '  Max value: $maxValue at position [$maxValueRow, $maxValueCol]',
        );

        // Print some values around the maximum
        print('  Values around maximum:');
        final startRow = math.max(0, maxValueRow - 2);
        final endRow = math.min(output[0].length - 1, maxValueRow + 2);
        final startCol = math.max(0, maxValueCol - 2);
        final endCol = math.min(output[0][0].length - 1, maxValueCol + 2);

        for (int i = startRow; i <= endRow; i++) {
          String rowValues = '';
          for (int j = startCol; j <= endCol; j++) {
            rowValues += '${output[0][i][j].toStringAsFixed(4)} ';
          }
          print('    Row $i: $rowValues');
        }
      }

      // Check first few detections
      print('  First 5 potential detections:');
      for (int i = 0; i < math.min(5, output[0][0].length); i++) {
        // Get box coordinates
        final x = output[0][0][i]; // center x
        final y = output[0][1][i]; // center y
        final w = output[0][2][i]; // width
        final h = output[0][3][i]; // height

        // Find max class probability
        double maxClassProb = 0;
        int maxClassIndex = -1;
        for (int c = 0; c < output[0].length - 4; c++) {
          if (output[0][c + 4][i] > maxClassProb) {
            maxClassProb = output[0][c + 4][i];
            maxClassIndex = c;
          }
        }

        print(
          '    Detection $i: box=[${x.toStringAsFixed(4)}, ${y.toStringAsFixed(4)}, ${w.toStringAsFixed(4)}, ${h.toStringAsFixed(4)}], class=$maxClassIndex, prob=${maxClassProb.toStringAsFixed(4)}',
        );
      }

      // Check if there are any high confidence detections
      int highConfCount = 0;
      for (int i = 0; i < output[0][0].length; i++) {
        double maxClassProb = 0;
        for (int c = 0; c < output[0].length - 4; c++) {
          if (output[0][c + 4][i] > maxClassProb) {
            maxClassProb = output[0][c + 4][i];
          }
        }
        if (maxClassProb > 0.1) {
          // Lower threshold for debugging
          highConfCount++;
        }
      }
      print('  Detections with confidence > 0.1: $highConfCount');
    } catch (e) {
      print('Error in output inspection: $e');
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

      // Inspect the output
      //inspectModelOutput(output);

      // Process the output to get detection boxes
      final results = processOutput(output, image.width, image.height);

      print('Results: $results');

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
          // Convert normalized coordinates to actual pixel coordinates
          final xmin = ((x - w / 2) * imageWidth).round();
          final ymin = ((y - h / 2) * imageHeight).round();
          final xmax = ((x + w / 2) * imageWidth).round();
          final ymax = ((y + h / 2) * imageHeight).round();

          // Ensure coordinates are within image bounds
          final boundedXmin = math.max(0, xmin);
          final boundedYmin = math.max(0, ymin);
          final boundedXmax = math.min(imageWidth, xmax);
          final boundedYmax = math.min(imageHeight, ymax);

          // Only add if the box has positive area
          if (boundedXmax > boundedXmin && boundedYmax > boundedYmin) {
            detections.add({
              'class': classIndex,
              'className':
                  _labels != null && classIndex < _labels!.length
                      ? _labels![classIndex]
                      : 'Unknown',
              'confidence': maxClassProb,
              'box': [boundedXmin, boundedYmin, boundedXmax, boundedYmax],
            });
          }
        }
      }

      print('Found ${detections.length} detections before NMS');

      // Apply non-maximum suppression to remove overlapping boxes
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
        final iou = _calculateIoU(boxes[i]['box'], boxes[j]['box']);

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

  void checkHighConfidenceDetection(List<dynamic> output) {
    try {
      // Check the detection at position 8194 (where we found max value)
      final i = 8194;

      // Get bounding box coordinates
      final x = output[0][0][i]; // center x
      final y = output[0][1][i]; // center y
      final w = output[0][2][i]; // width
      final h = output[0][3][i]; // height

      print('High confidence detection at index $i:');
      print('  Box coordinates: x=$x, y=$y, w=$w, h=$h');

      // Check class probabilities
      print('  Class probabilities:');
      List<double> classProbs = [];
      for (int c = 0; c < 10; c++) {
        // Just print first 10 classes
        classProbs.add(output[0][c + 4][i]);
      }
      print('  First 10 classes: $classProbs');

      // Find max class probability
      double maxClassProb = 0;
      int maxClassIndex = -1;
      for (int c = 0; c < output[0].length - 4; c++) {
        if (output[0][c + 4][i] > maxClassProb) {
          maxClassProb = output[0][c + 4][i];
          maxClassIndex = c;
        }
      }
      print('  Max class: $maxClassIndex, probability: $maxClassProb');
    } catch (e) {
      print('Error checking high confidence detection: $e');
    }
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
        ],
      ),
    );
  }
}
