import 'dart:math' as math;

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
      print('  Max value: $maxValue at position [$maxValueRow, $maxValueCol]');

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

// Future<void> testWithSampleImage(String imageFileUrl) async {
//   try {
//     // Load image from the file path
//     final ByteData bytes = await rootBundle.load(
//       'yolo11n_saved_model/assets/test.jpg',
//     );
//     final List<int> list = bytes.buffer.asUint8List();

//     // Create a temporary file
//     final tempDir = await getTemporaryDirectory();
//     final tempFile = File('${tempDir.path}/test_byte.jpg');
//     await tempFile.writeAsBytes(list);

//     // Run detection
//     final detections = await runDetection(tempFile);

//     print('Test image detections: $detections');
//   } catch (e) {
//     print('Error testing with sample image: $e');
//   }
// }
