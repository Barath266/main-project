import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:typed_data';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Image Classification',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: ImageAnalyze(),
    );
  }
}

class ImageAnalyze extends StatefulWidget {
  const ImageAnalyze({Key? key}) : super(key: key);

  @override
  State<ImageAnalyze> createState() => _ImageAnalyzeState();
}

class _ImageAnalyzeState extends State<ImageAnalyze> {
  Uint8List? _imageBytes;
  File? _imageFile;
  bool _loading = false;
  String _selectedModel = 'choosemodel'; // Initially selected value
  List<String> _modelList = ['vit', 'pcawithcnn', 'hybrid'];

  Future<void> _pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: false,
    );

    if (result != null) {
      setState(() {
        _loading = true;
        if (kIsWeb) {
          _imageBytes = result.files.single.bytes;
        } else {
          _imageFile = File(result.files.single.path!);
        }
      });
    } else {
      // User canceled the picker
      print('No image selected.');
    }
  }

  Future<void> _predict() async {
    if (_selectedModel == 'choosemodel' || _imageFile == null) {
      // Model or image not selected
      return;
    }

    setState(() {
      _loading = true;
    });

    String url =
        'http://192.168.40.90:5000/predict'; // Replace with your backend URL

    // Prepare the request body
    var request = http.MultipartRequest('POST', Uri.parse(url));
    request.files
        .add(await http.MultipartFile.fromPath('image', _imageFile!.path));
    request.fields['model'] = _selectedModel;

    // Send the request
    try {
      var response = await request.send();
      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        // Parse the response data
        var predictionResult = json.decode(responseData);
        String predictedCategory = predictionResult['predicted_category'];
        // Update UI with the predicted category
        setState(() {
          _loading = false;
          // Show the predicted category
          _showPredictionResult(predictedCategory);
        });
      } else {
        // Handle error
        print('Failed to receive response from server.');
      }
    } catch (e) {
      // Handle network error
      print('Error: $e');
      setState(() {
        _loading = false;
      });
    }
  }

  void _showPredictionResult(String predictedCategory) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: Text('Prediction Result'),
          content: Text('Predicted Category: $predictedCategory'),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.of(context).pop();
              },
              child: Text('OK'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(
          "Analyze",
          style: TextStyle(
            fontSize: 24,
            color: Color.fromARGB(255, 66, 96, 32),
            fontWeight: FontWeight.bold,
          ),
        ),
      ),
      body: SingleChildScrollView(
        child: Container(
          color: Colors.white,
          alignment: Alignment.center,
          padding: EdgeInsets.all(10),
          child: Column(
            children: [
              Container(
                padding: EdgeInsets.symmetric(horizontal: 35, vertical: 40),
                child: Container(
                  alignment: Alignment.center,
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        height: 400, // Medium size
                        width: 400, // Medium size
                        child: Center(
                          child: _imageBytes == null && _imageFile == null
                              ? const Text(
                                  'No image selected',
                                  style: TextStyle(
                                    color: Colors.black,
                                    fontSize: 18,
                                    fontWeight: FontWeight.w400,
                                  ),
                                )
                              : _imageBytes != null
                                  ? Image.memory(
                                      _imageBytes!,
                                      fit: BoxFit.contain,
                                    )
                                  : Image.file(
                                      _imageFile!,
                                      fit: BoxFit.contain,
                                    ),
                        ),
                      ),
                      SizedBox(height: 20),
                      ElevatedButton(
                        onPressed: _pickImage,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Color.fromARGB(255, 43, 97, 45),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(15),
                          ),
                        ),
                        child: Padding(
                          padding: EdgeInsets.symmetric(
                              horizontal: 24, vertical: 17),
                          child: Text(
                            'Select Image',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                            ),
                          ),
                        ),
                      ),
                      SizedBox(height: 20),
                      DropdownButton<String>(
                        value: _selectedModel,
                        items: [
                          DropdownMenuItem<String>(
                            value: 'choosemodel',
                            child: Text('Choose Model'),
                          ),
                          ..._modelList.map((String model) {
                            return DropdownMenuItem<String>(
                              value: model,
                              child: Text(model),
                            );
                          }),
                        ],
                        onChanged: (String? newValue) {
                          if (newValue != null) {
                            setState(() {
                              _selectedModel = newValue;
                            });
                          }
                        },
                      ),
                      SizedBox(height: 20),
                      ElevatedButton(
                        onPressed: _predict,
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Color.fromARGB(255, 43, 97, 45),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(15),
                          ),
                        ),
                        child: Padding(
                          padding: EdgeInsets.symmetric(
                              horizontal: 24, vertical: 17),
                          child: Text(
                            'Predict',
                            style: TextStyle(
                              color: Colors.white,
                              fontSize: 16,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
