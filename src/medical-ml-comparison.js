import '@marcellejs/core/dist/marcelle.css';
import {
  datasetBrowser,
  button,
  dashboard,
  classificationPlot,
  trainingPlot,
} from '@marcellejs/core';
import { instances, source, sourceImages, store } from './common';
import {
  createModel,
  createPredictionStream,
  managerControls,
  setupBatchPrediction,
  setupTestSet,
} from './ml-utils';
import { runManager } from './modules';

const dash = dashboard({
  title: 'Marcelle: Skin Lesion Classification',
  author: 'Louise',
});

const numModels = 2;
const modelIdx = Array.from(Array(numModels), (_, i) => i);

const classifiers = modelIdx.map((i) => {
  const classifier = createModel(); // .sync(`skin-lesion-classifier-${i}`);
  classifier.title = `Classifier ${i + 1}`;
  return classifier;
});

// -----------------------------------------------------------
// Training Experiment Management
// -----------------------------------------------------------

const managers = modelIdx.map(() => runManager({ dataStore: store }));

const controls = managers.map((manager, i) => managerControls(manager, classifiers[i]));
const runSelectors = controls.map(({ selectRun }) => selectRun);
const modelSelectors = controls.map(({ selectModel }) => selectModel);
const modelLoaders = controls.map(({ loadModel }) => loadModel);

const lossPlots = managers.map((manager) =>
  trainingPlot(manager, {
    loss: ['loss', 'val_loss'],
  }),
);
const accuracyPlots = managers.map((manager) =>
  trainingPlot(manager, {
    accuracy: ['accuracy', 'val_accuracy'],
  }),
);

// -----------------------------------------------------------
// REAL-TIME PREDICTION
// -----------------------------------------------------------

const predictionStreams = classifiers.map((classifier) =>
  createPredictionStream(
    source.$images.filter(() => dash.$page.value === 'real-time-testing'),
    classifier,
  ),
);

const predictionPlots = predictionStreams.map((p, i) => {
  const x = classificationPlot(p);
  x.title = `Predictions (Model ${i + 1})`;
  return x;
});

// -----------------------------------------------------------
// BATCH PREDICTION
// -----------------------------------------------------------

const { testSet, selectLabel } = setupTestSet(
  instances.filter(() => dash.$page.value === 'batch-testing'),
);
const testSetBrowser = datasetBrowser(testSet);

const predictButton = button({ text: 'Update predictions' });
predictButton.title = 'Batch predictions';

const confusionMatrices = classifiers.map((classifier, i) => {
  const { confMat } = setupBatchPrediction(testSet, classifier, predictButton);
  confMat.title = `Model ${i + 1}: Confusion Matrix`;
  return confMat;
});

// -----------------------------------------------------------
// DASHBOARDS
// -----------------------------------------------------------

dash
  .page('Model Selector')
  .use(runSelectors, modelSelectors, modelLoaders, classifiers)
  .use(lossPlots, accuracyPlots);
// .use(trainingPlots, runSummary, modelSummary);
dash.page('Real-time Testing').useLeft(source, sourceImages).use(predictionPlots, classifiers);
dash
  .page('Batch Testing')
  .useLeft(source, selectLabel)
  .use(testSetBrowser, classifiers, predictButton, confusionMatrices);

dash.settings
  .dataStores(store)
  .datasets(testSet)
  .models(...classifiers);

dash.start();
