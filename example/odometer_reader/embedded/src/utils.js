const ort = require('onnxruntime-web');
const sharp = require('sharp');
const fs = require('fs');

async function loadModel(model_path) {
    try {
        const session = await ort.InferenceSession.create(model_path);
        console.log('Model loaded successfully');
        return session;
    } catch (error) {
        console.error('Failed to load model:', error);
        throw error;
    }
}



async function preprocessImage(imagePath, width, height) {
    try {
        if (!fs.existsSync(imagePath)) {
            throw new Error(`Image file not found: ${imagePath}`);
        }
  
        const preprocessed = await sharp(imagePath)
            .resize(width, height)
            .grayscale()
            .raw()
            .toBuffer();
  
        // Simple normalization to [-1,1]
        const preprocessedData = new Float32Array(height * width);
        for (let i = 0; i < preprocessed.length; i++) {
            preprocessedData[i] = (preprocessed[i] - 127.5) / 127.5;
          }
  
        // Prepare tensor data
        const tensorData = new Float32Array(1 * 1 * height * width);
        for (let h = 0; h < height; h++) {
            for (let w = 0; w < width; w++) {
                tensorData[h * width + w] = preprocessedData[h * width + w];
            }
        }
        return tensorData;
    } catch (error) {
        console.error('Error preprocessing image:', error);
        throw error;
    }
  }
  






module.exports = { loadModel, preprocessImage };