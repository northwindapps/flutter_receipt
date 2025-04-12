import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
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

      final detections = await runDetection(imageFile);

      print(detections);
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

    List<dynamic> detections = [];
    // Process output to get bounding boxes, classes, and confidence scores
    // ...

    return detections;
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
