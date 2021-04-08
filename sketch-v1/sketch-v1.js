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
  trainingProgress,
} from '@marcellejs/core';

// Main components
const input = sketchpad();
const featureExtractor = mobilenet();
const store = dataStore({ location: 'localStorage' });
const trainingSet = dataset({ name: 'TrainingSet', dataStore: store });
const classifier = mlp({ layers: [64, 32], epochs: 20, dataStore: store });
classifier.sync('sketch-classifier');

// Additional widgets and visualizations
const classLabel = textfield();
const captureButton = button({ text: 'Capture this drawing' });
const trainButton = button({ text: 'Train the classifier' });
const predictButton = button({ text: 'Predict label' });

const trainingSetBrowser = datasetBrowser(trainingSet);
const progress = trainingProgress(classifier);

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

// Prediction Pipeline
const $features = predictButton.$click
  .sample(input.$images)
  .map((imgData) => featureExtractor.process(imgData))
  .awaitPromises();
//
const $predictions = $features.map((features) => classifier.predict(features)).awaitPromises();
//
const predictionViz = classificationPlot($predictions);

// Dashboard definition
const myDashboard = dashboard({ title: 'Sketch App (v1)', author: 'Suzanne' });

myDashboard
  .page('Main')
  .useLeft(input, featureExtractor)
  .use([classLabel, captureButton], trainingSetBrowser, trainButton, progress, [
    predictButton,
    predictionViz,
  ]);

myDashboard.settings.dataStores(store).datasets(trainingSet).models(classifier);

myDashboard.start();
