import '@marcellejs/core/dist/marcelle.css';
import {
  datasetBrowser,
  button,
  dashboard,
  confidencePlot,
  trainingHistory,
} from '@marcellejs/core';
import { instances, source, sourceImages, store } from './common';
import {
  createModel,
  createPredictionStream,
  managerControls,
  setupBatchPrediction,
  setupTestSet,
} from './ml-utils';

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

const hist = trainingHistory(store, {
  metrics: ['accuracy', 'val_accuracy'],
  actions: [
    { name: 'select model 1', multiple: false },
    { name: 'select model 2', multiple: false },
  ],
});

const controls = classifiers.map((classifier, i) =>
  managerControls(hist, classifier, `select model ${i + 1}`),
);
const checkpointSelectors = controls.map(({ selectCheckpoint }) => selectCheckpoint);
const modelLoaders = controls.map(({ loadModel }) => loadModel);

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
  const x = confidencePlot(p);
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

dash.page('Model Selector').use(hist, checkpointSelectors, modelLoaders, classifiers);
dash.page('Real-time Testing').sidebar(source, sourceImages).use(predictionPlots, classifiers);
dash
  .page('Batch Testing')
  .sidebar(source, selectLabel)
  .use(testSetBrowser, classifiers, predictButton, confusionMatrices);

dash.settings
  .dataStores(store)
  .datasets(testSet)
  .models(...classifiers);

dash.show();
