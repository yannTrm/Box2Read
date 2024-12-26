const ort = require('onnxruntime-web');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const { beamSearchDecode } = require('./ctc_decode');
const { loadModel, preprocessImage } = require('./utils')

const MODEL_PATH = './models/odometer_reader.onnx';
const TARGET_HEIGHT = 32;
const TARGET_WIDTH = 100;


async function runInference(session, preprocessedImage) {
  try {
      const tensor = new ort.Tensor('float32', preprocessedImage, [1, 1, TARGET_HEIGHT, TARGET_WIDTH]);
      const inputName = session.inputNames[0];
      const feeds = {};
      feeds[inputName] = tensor;

      console.log('Running inference...');
      const outputData = await session.run(feeds);
      const outputName = session.outputNames[0];
      const output = outputData[outputName];
      
      // Reshape output to [25, 11] (time steps x classes)
      const logits = Array.from(output.data);
      const timeSteps = 25;
      const numClasses = 11;
      const reshapedLogits = [];
      
      // Verify dimensions
      console.log('Original output shape:', output.dims);
      console.log('Expected shape: [25, 11]');
      
      // Reshape and apply log softmax per timestep
      for (let t = 0; t < timeSteps; t++) {
        const timeStep = logits.slice(t * numClasses, (t + 1) * numClasses);
        
        // Apply log softmax
        const maxLogit = Math.max(...timeStep);
        const expSum = Math.log(
            timeStep
                .map(x => Math.exp(x - maxLogit))
                .reduce((a, b) => a + b, 0)
        );
        const logProbs = timeStep.map(x => (x - maxLogit) - expSum);
        
        reshapedLogits.push(logProbs);
        
        // Debug first few timesteps
        if (t < 2) {
            console.log(`\nTimestep ${t} logits:`, logProbs);
        }
    }

      // Apply beam search on correctly shaped input
      const decoded = beamSearchDecode(reshapedLogits, 0, 10);
      const label2char = {0: ' ', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 
                        6: '5', 7: '6', 8: '7', 9: '8', 10: '9'};
      const prediction = decoded.map(label => label2char[label]).join('');
      
      console.log('\nFinal prediction:', prediction);
      return prediction;
  } catch (error) {
      console.error('Error during inference:', error);
      throw error;
  }
}

async function processDirectory(inputDir, outputDir) {
  try {
      // Create output directory if it doesn't exist
      if (!fs.existsSync(outputDir)) {
          fs.mkdirSync(outputDir, { recursive: true });
      }

      // Load model once for all images
      const model = await loadModel(MODEL_PATH);
      
      // Get all PNG and JPG files
      const files = fs.readdirSync(inputDir)
          .filter(file => file.match(/\.(jpg|jpeg|png)$/i));
      
      console.log(`Found ${files.length} images to process`);

      // Process each image
      for (const file of files) {
          console.log(`\nProcessing ${file}...`);
          const imagePath = path.join(inputDir, file);
          
          try {
              const preprocessedImage = await preprocessImage(imagePath, TARGET_WIDTH, TARGET_HEIGHT);
              const prediction = await runInference(model, preprocessedImage);
              
              // Create new filename with prediction
              const ext = path.extname(file);
              const basename = path.basename(file, ext);
              const newFilename = `${basename}_pred_${prediction}${ext}`;
              
              // Copy original image to output dir with new name
              fs.copyFileSync(
                  imagePath, 
                  path.join(outputDir, newFilename)
              );
              
              console.log(`Saved as: ${newFilename}`);
          } catch (error) {
              console.error(`Error processing ${file}:`, error);
              continue;
          }
      }
  } catch (error) {
      console.error('Error processing directory:', error);
      throw error;
  }
}

async function main() {
  try {
      const inputDir = './images/odometer';  // Update this path
      const outputDir = './images/odometer_predictions';
      await processDirectory(inputDir, outputDir);
      console.log('\nProcessing complete');
  } catch (error) {
      console.error('Error:', error);
  }
}

main();

