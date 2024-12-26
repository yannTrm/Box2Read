const ort = require('onnxruntime-web');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');

const { loadModel } = require('./utils');

const MODEL_PATH = './models/milestone-box-detector.onnx';
const IMG_SIZE = 640;


const { createCanvas, loadImage } = require('canvas');


// Préparer l'image d'entrée
async function preprocess(imagePath, inputSize = 640) {
    const image = await loadImage(imagePath);
    const originalWidth = image.width;
    const originalHeight = image.height;

    // Redimensionner l'image dans un canvas
    const canvas = createCanvas(inputSize, inputSize);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, inputSize, inputSize);

    // Extraire les données d'image
    const imageData = ctx.getImageData(0, 0, inputSize, inputSize).data;

    // Convertir en tableau Float32Array et réorganiser en format (C, H, W)
    const floatData = new Float32Array(inputSize * inputSize * 3);
    for (let i = 0; i < inputSize * inputSize; i++) {
        const pixelIndex = i * 4; // (R, G, B, A)
        floatData[i] = imageData[pixelIndex] / 255.0; // R
        floatData[inputSize * inputSize + i] = imageData[pixelIndex + 1] / 255.0; // G
        floatData[2 * inputSize * inputSize + i] = imageData[pixelIndex + 2] / 255.0; // B
    }

    return { floatData, originalWidth, originalHeight };
}


// Effectuer l'inférence
async function runInference(session, inputData, inputSize = 640) {
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
    fs.writeFileSync('output.png', outputBuffer);
    console.log('Bounding boxes drawn and saved as output.png');
}


// Trier les boxes par confiance et récupérer les 5 premières
function getTopPredictions(boxes, topN = 5) {
    return boxes
        .sort((a, b) => b.confidence - a.confidence) // Trier par confiance décroissante
        .slice(0, topN); // Récupérer les N premières
}

// Pipeline principal
(async () => {
    const modelPath = './models/milestone-box-detector.onnx';
    const imagePath = './images/yolo/scraped_0bdjWL_1654866810755.jpg';



    try {
        const session = await loadModel(modelPath);
        const { floatData, originalWidth, originalHeight } = await preprocess(imagePath);
        const output = await runInference(session, { floatData }, 640);
        const boxes = postProcess(output, 640, originalWidth, originalHeight);

        if (boxes.length > 0) {
            const topPredictions = getTopPredictions(boxes, 5);
            console.log('Top 5 Predictions:', topPredictions);

            await drawBoundingBoxes(imagePath, topPredictions);
        } else {
            console.log('No predictions above the confidence threshold.');
        }
    } catch (error) {
        console.error('Error during inference pipeline:', error);
    }
})();
