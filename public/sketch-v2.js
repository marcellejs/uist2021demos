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
  notification,
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
trainingSet.$changes.subscribe((changes) => {
  if (changes.length === 0 || changes[0].level === 'dataset') return;
  if (trainingSet.$labels.value.length < 2) {
    notification({
      title: 'Tip',
      message: 'You need to have at least two classes to train the model',
      duration: 5000,
    });
  } else if (
    Object.values(trainingSet.$classes.value)
      .map((x) => x.length)
      .reduce((x, y) => x || y < 2, false)
  ) {
    notification({
      title: 'Tip',
      message: 'You need to record at least two examples per class',
      duration: 5000,
    });
  } else {
    classifier.train(trainingSet);
  }
});

// Prediction Pipeline (Online)
const $features = input.$images.map((imgData) => featureExtractor.process(imgData)).awaitPromises();

const $trainingSuccess = classifier.$training.filter((x) => x.status === 'success');

const $predictions = $features
  .merge($trainingSuccess.sample($features))
  .map((features) => classifier.predict(features))
  .awaitPromises();

const predictionViz = classificationPlot($predictions);

$predictions.subscribe(({ label }) => {
  classLabel.$text.set(label);
});

// Dashboard definition
const myDashboard = dashboard({ title: 'Sketch App (v2)', author: 'Suzanne' });

myDashboard
  .page('Main')
  .useLeft(input, featureExtractor)
  .use(predictionViz, [classLabel, captureButton], progress, trainingSetBrowser);

myDashboard.settings.dataStores(store).datasets(trainingSet).models(classifier);

myDashboard.start();

// Help messages

input.$images
  .filter(() => trainingSet.$count.value === 0 && !classifier.ready)
  .take(1)
  .subscribe(() => {
    notification({
      title: 'Tip',
      message: 'Start by editing the label and adding the drawing to the dataset',
      duration: 5000,
    });
  });
