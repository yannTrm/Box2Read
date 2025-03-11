const ort = require('onnxruntime-web');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

const { beamSearchDecode } = require('./ctc_decode');
const { loadModel } = require('./utils');

const BOX_MODEL_PATH = './models/milestone-box-detector.onnx';
const OCR_MODEL_PATH = './models/odometer_reader.onnx';
const IMG_SIZE = 640;
const TARGET_HEIGHT = 32;
const TARGET_WIDTH = 100;

// Préparer l'image d'entrée pour la détection de box
async function preprocess(imagePath, inputSize = 640) {
    const image = await loadImage(imagePath);
    const originalWidth = image.width;
    const originalHeight = image.height;

    const canvas = createCanvas(inputSize, inputSize);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, inputSize, inputSize);

    const imageData = ctx.getImageData(0, 0, inputSize, inputSize).data;

    const floatData = new Float32Array(inputSize * inputSize * 3);
    for (let i = 0; i < inputSize * inputSize; i++) {
        const pixelIndex = i * 4;
        floatData[i] = imageData[pixelIndex] / 255.0;
        floatData[inputSize * inputSize + i] = imageData[pixelIndex + 1] / 255.0;
        floatData[2 * inputSize * inputSize + i] = imageData[pixelIndex + 2] / 255.0;
    }

    return { floatData, originalWidth, originalHeight };
}

// Effectuer l'inférence pour la détection de box
async function runBoxInference(session, inputData, inputSize = 640) {
    const inputTensor = new ort.Tensor('float32', inputData.floatData, [1, 3, inputSize, inputSize]);
    const feeds = { [session.inputNames[0]]: inputTensor };

    const results = await session.run(feeds);
    const output = results[session.outputNames[0]].data;
    return output;
}

// Post-traitement pour extraire les bounding boxes
function postProcess(output, inputSize, originalWidth, originalHeight, confidenceThreshold = 0.001) {
    const numDetections = output.length / 5;
    const stride = numDetections;

    const xCenters = output.slice(0, stride);
    const yCenters = output.slice(stride, stride * 2);
    const widths = output.slice(stride * 2, stride * 3);
    const heights = output.slice(stride * 3, stride * 4);
    const confidences = output.slice(stride * 4, stride * 5);

    const boxes = [];

    for (let i = 0; i < numDetections; i++) {
        if (confidences[i] > confidenceThreshold) {
            const scaledX = xCenters[i] * originalWidth / inputSize;
            const scaledY = yCenters[i] * originalHeight / inputSize;
            const scaledW = widths[i] * originalWidth / inputSize;
            const scaledH = heights[i] * originalHeight / inputSize;
            boxes.push({ x: scaledX, y: scaledY, w: scaledW, h: scaledH, confidence: confidences[i] });
        }
    }
    return boxes;
}

// Prétraitement de l'image pour l'OCR
async function preprocessImage(buffer, width, height) {
    const preprocessed = await sharp(buffer)
        .resize(width, height)
        .grayscale()
        .raw()
        .toBuffer();

    const preprocessedData = new Float32Array(height * width);
    for (let i = 0; i < preprocessed.length; i++) {
        preprocessedData[i] = (preprocessed[i] - 127.5) / 127.5;
    }

    const tensorData = new Float32Array(1 * 1 * height * width);
    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            tensorData[h * width + w] = preprocessedData[h * width + w];
        }
    }
    return tensorData;
}

// Effectuer l'inférence pour l'OCR
async function runOCRInference(session, preprocessedImage) {
    const tensor = new ort.Tensor('float32', preprocessedImage, [1, 1, TARGET_HEIGHT, TARGET_WIDTH]);
    const inputName = session.inputNames[0];
    const feeds = {};
    feeds[inputName] = tensor;

    const outputData = await session.run(feeds);
    const outputName = session.outputNames[0];
    const output = outputData[outputName];

    const logits = Array.from(output.data);
    const timeSteps = 25;
    const numClasses = 11;
    const reshapedLogits = [];

    for (let t = 0; t < timeSteps; t++) {
        const timeStep = logits.slice(t * numClasses, (t + 1) * numClasses);
        const maxLogit = Math.max(...timeStep);
        const expSum = Math.log(
            timeStep.map(x => Math.exp(x - maxLogit)).reduce((a, b) => a + b, 0)
        );
        const logProbs = timeStep.map(x => (x - maxLogit) - expSum);
        reshapedLogits.push(logProbs);
    }

    const decoded = beamSearchDecode(reshapedLogits, 0, 10);
    const label2char = {0: ' ', 1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9'};
    const prediction = decoded.map(label => label2char[label]).join('');

    return prediction;
}

// Dessiner les bounding boxes sur l'image
async function drawBoundingBoxes(imagePath, boxes) {
    const image = await loadImage(imagePath);
    const canvas = createCanvas(image.width, image.height);
    const ctx = canvas.getContext('2d');

    ctx.drawImage(image, 0, 0);

    ctx.strokeStyle = 'green';
    ctx.lineWidth = 2;
    ctx.font = '16px Arial';
    ctx.fillStyle = 'green';

    boxes.forEach(box => {
        const x = box.x - box.w / 2;
        const y = box.y - box.h / 2;
        ctx.strokeRect(x, y, box.w, box.h);
        ctx.fillText(box.confidence.toFixed(2), x, y - 5);
    });

    const outputBuffer = canvas.toBuffer('image/png');
    fs.writeFileSync('output_with_boxes.png', outputBuffer);
    console.log('Bounding boxes drawn and saved as output_with_boxes.png');
}

// Pipeline principal
(async () => {
    const imagePath = './images/yolo/scraped_0EHYLJ_1654870976654.jpg';

    try {
        const boxSession = await loadModel(BOX_MODEL_PATH);
        const ocrSession = await loadModel(OCR_MODEL_PATH);

        const { floatData, originalWidth, originalHeight } = await preprocess(imagePath);
        const boxOutput = await runBoxInference(boxSession, { floatData }, IMG_SIZE);
        const boxes = postProcess(boxOutput, IMG_SIZE, originalWidth, originalHeight);

        if (boxes.length > 0) {
            const topPredictions = boxes.sort((a, b) => b.confidence - a.confidence).slice(0, 5);

            for (const box of topPredictions) {
                const x = Math.max(0, box.x - box.w / 2);
                const y = Math.max(0, box.y - box.h / 2);
                const width = Math.min(originalWidth, box.w);
                const height = Math.min(originalHeight, box.h);

                const croppedImageBuffer = await sharp(imagePath)
                    .extract({ left: Math.round(x), top: Math.round(y), width: Math.round(width), height: Math.round(height) })
                    .toBuffer();

                const preprocessedImage = await preprocessImage(croppedImageBuffer, TARGET_WIDTH, TARGET_HEIGHT);
                const prediction = await runOCRInference(ocrSession, preprocessedImage);
                console.log(`Prediction for box [${box.x}, ${box.y}, ${box.w}, ${box.h}]: ${prediction}`);
            }

            await drawBoundingBoxes(imagePath, topPredictions);
        } else {
            console.log('No predictions above the confidence threshold.');
        }
    } catch (error) {
        console.error('Error during inference pipeline:', error);
    }
})();