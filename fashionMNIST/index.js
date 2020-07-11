async function runExample() {
  // Create an ONNX inference session with WebGL backend.
  const session = new onnx.InferenceSession({ backendHint: 'webgl' });

  // Load an ONNX model. This model is Resnet50 that takes a 1*3*224*224 image and classifies it.
  await session.loadModel("./fashionMNIST.onnx");

  // Load image.
  const imageLoader = new ImageLoader(imageSize, imageSize);
  const imageData = await imageLoader.getImageData('./test.jpg');

  // Preprocess the image data to match input dimension requirement, which is 1*3*224*224.
  const width = imageSize;
  const height = imageSize;
  const preprocessedData = preprocess(imageData.data, width, height);

  const inputTensor = new onnx.Tensor(preprocessedData, 'float32', [1, 1, width, height]);
  // Run model with Tensor inputs and get the result.
  const outputMap = await session.run([inputTensor]);
  const outputData = outputMap.values().next().value.data;

  const maxPrediction = Math.max(...outputData);

  console.log(outputData.length);
  console.log(outputData);
  console.log(maxPrediction);

  const element = document.getElementById('predictions');
  element.innerHTML = '';
  const results = [];
  for (let i = 0; i < outputData.length; i++) {
    results.push(`${fashionMNISTClasses[i][0]}: ${outputData[i] * 100}%`);
  }  
  element.innerHTML = results.join('<br/>');

  // Render the output result in html.
  // printMatches(outputData);
}

var fashionMNISTClasses = {
  '0': ['T-shirt/top'],
  '1': ['Trouser'],
  '2': ['Pullover'],
  '3': ['Dress'],
  '4': ['Coat'],
  '5': ['Sandal'],
  '6': ['Shirt'],
  '7': ['Sneaker'],
  '8': ['Bag'],
  '9': ['Ankle'],
};

/**
 * Preprocess raw image data to match Resnet50 requirement.
 */
function preprocess(data, width, height) {
  const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
  const dataProcessed = ndarray(new Float32Array(width * height * 1), [1, 1, height, width]);

  // Normalize 0-255 to (-1)-1
  ndarray.ops.divseq(dataFromImage, 128.0);
  ndarray.ops.subseq(dataFromImage, 1.0);

  // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
  ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
  ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
  ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));

  return dataProcessed.data;
}