/* eslint-disable import/extensions */
import '@marcellejs/core/dist/marcelle.css';
import {
  datasetBrowser,
  button,
  dashboard,
  dataset,
  dataStore,
  mlp,
  mobilenet,
  classificationPlot,
  sketchpad,
  textfield,
  toggle,
  parameters,
  trainingProgress,
  trainingPlot,
  batchPrediction,
  confusionMatrix,
  throwError,
} from '@marcellejs/core';

// Main components
const input = sketchpad();
const featureExtractor = mobilenet();
const store = dataStore({ location: 'localStorage' });
const trainingSet = dataset({ name: 'TrainingSet', dataStore: store });
const classifier = mlp({ layers: [64, 32], epochs: 20, dataStore: store });
classifier.sync('sketch-classifier');
const batchResults = batchPrediction({ name: 'mlp', dataStore: store });

// Additional widgets and visualizations
const classLabel = textfield();
const captureButton = button({ text: 'Capture this drawing' });
const trainButton = button({ text: 'Train the classifier' });
const batchPredictButton = button({ text: 'Update batch predictions on the training dataset' });
const realTimePredictToggle = toggle({ text: 'Toggle real-time prediction' });

const trainingSetBrowser = datasetBrowser(trainingSet);
const classifierParams = parameters(classifier);
const progress = trainingProgress(classifier);
const lossCurves = trainingPlot(classifier);
const confusion = confusionMatrix(batchResults);

// Dataset Pipeline
const $instances = captureButton.$click
  .sample(input.$images.zip((thumbnail, data) => ({ thumbnail, data }), input.$thumbnails))
  .map(async (instance) => ({
    ...instance,
    type: 'sketch',
    label: classLabel.$text.value,
    features: await featureExtractor.process(instance.data),
  }))
  .awaitPromises();

trainingSet.capture($instances);

// Training Pipeline
trainButton.$click.subscribe(() => classifier.train(trainingSet));

// Real-time Prediction Pipeline
const $features = input.$images
  .filter(() => realTimePredictToggle.$checked.value && classifier.ready)
  .map((img) => featureExtractor.process(img))
  .awaitPromises();

const $predictions = $features.map((features) => classifier.predict(features)).awaitPromises();

const predictionViz = classificationPlot($predictions);

// Batch Prediction Pipeline
batchPredictButton.$click.subscribe(async () => {
  if (!classifier.ready) {
    throwError(new Error('No classifier has been trained'));
  }
  await batchResults.clear();
  await batchResults.predict(classifier, trainingSet);
});

// Dashboard definition
const myDashboard = dashboard({ title: 'Sketch App (v1++)', author: 'Suzanne' });

myDashboard
  .page('Data Management')
  .useLeft(input, featureExtractor)
  .use([classLabel, captureButton], trainingSetBrowser);
myDashboard.page('Training').use(classifierParams, trainButton, progress, lossCurves);
myDashboard.page('Batch Prediction').use(batchPredictButton, confusion);
myDashboard.page('Real-time Prediction').useLeft(input).use(realTimePredictToggle, predictionViz);

myDashboard.settings.dataStores(store).datasets(trainingSet).models(classifier);

myDashboard.start();
