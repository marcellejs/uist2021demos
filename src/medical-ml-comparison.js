import '@marcellejs/core/dist/marcelle.css';
import {
  batchPrediction,
  datasetBrowser,
  button,
  confusionMatrix,
  dashboard,
  dataset,
  classificationPlot,
  select,
  tfGenericModel,
  trainingPlot,
} from '@marcellejs/core';
import { pythonTrained } from './modules';
import { instances, labels, source, sourceImages, store } from './common';

let $dashboardPage;

const numModels = 2;
const modelIdx = Array.from(Array(numModels), (_, i) => i);

const classifiers = modelIdx.map((i) => {
  const classifier = tfGenericModel({
    inputType: 'image',
    taskType: 'classification',
    dataStore: store,
  }).sync(`skin-lesion-classifier-${i}`);
  classifier.title = `Classifier ${i + 1}`;
  classifier.labels = labels;
  return classifier;
});

// -----------------------------------------------------------
// PYTHON-TRAINED MODULE
// -----------------------------------------------------------

const runSelectors = modelIdx.map(() => {
  const s = select();
  s.title = 'Select Training Run:';
  return s;
});

const modelSelectors = modelIdx.map(() => {
  const s = select();
  s.title = 'Select Model Version:';
  return s;
});

const monitors = modelIdx.map((i) => {
  const monitor = pythonTrained({ dataStore: store });
  monitor.$runs.subscribe((runs) => {
    runSelectors[i].$options.set(runs);
    if (!runSelectors[i].$value.value) {
      runSelectors[i].$value.set(runs[0]);
    }
  });
  runSelectors[i].$value.subscribe(monitor.loadRun.bind(monitor));
  monitor.$checkpoints.subscribe((checkpoint) => {
    modelSelectors[i].$options.set(
      checkpoint.map((x) => (x.epoch === 'final' ? 'Final' : `Epoch: ${x.epoch}`)),
    );
  });
  return monitor;
});

const modelLoaders = modelIdx.map((i) => {
  const s = button({ text: 'Load Model' });
  s.title = 'Load Model Version';
  s.$click.sample(modelSelectors[i].$value).subscribe((checkpoint) => {
    if (checkpoint) {
      const epoch = checkpoint === 'Final' ? 'final' : checkpoint.split('Epoch: ')[1];
      const modelPath = monitors[i].$checkpoints.value.filter(
        (x) => x.epoch.toString() === epoch,
      )[0].url;
      const url = `${store.location}/tfjs-models/${modelPath}`;
      // eslint-disable-next-line no-console
      console.log('Loaded model:', url);
      classifiers[i].loadFromUrl(url);
    }
  });
  return s;
});

const lossPlots = modelIdx.map((i) =>
  trainingPlot(monitors[i], {
    loss: ['loss', 'val_loss'],
  }),
);
const accuracyPlots = modelIdx.map((i) =>
  trainingPlot(monitors[i], {
    accuracy: ['accuracy', 'val_accuracy'],
  }),
);

// -----------------------------------------------------------
// REAL-TIME PREDICTION
// -----------------------------------------------------------

const predictionStreams = modelIdx.map((i) =>
  source.$images
    .filter(() => $dashboardPage.value === 'real-time-testing')
    .map(async (img) => classifiers[i].predict(img))
    .awaitPromises(),
);

const predictionPlots = predictionStreams.map((p, i) => {
  const x = classificationPlot(p);
  x.title = `Predictions (Model ${i + 1})`;
  return x;
});

// -----------------------------------------------------------
// CAPTURE TO DATASET
// -----------------------------------------------------------

const trainingSet = dataset({ name: 'TrainingSet', dataStore: store });

const label = select({ options: labels });
label.title = 'Select label:';
trainingSet.capture(
  instances
    .filter(() => $dashboardPage.value === 'batch-testing')
    .map((y) => ({ ...y, label: label.$value.value })),
);

const trainingSetBrowser = datasetBrowser(trainingSet);

// -----------------------------------------------------------
// BATCH PREDICTION
// -----------------------------------------------------------

const batchPredictions = modelIdx.map((i) =>
  batchPrediction({ name: `skin-predictions-${i}`, dataStore: store }),
);
const predictButton = button({ text: 'Update predictions' });
const confusionMatrices = modelIdx.map((i) => {
  const c = confusionMatrix(batchPredictions[i]);
  c.title = `Model ${i + 1}: Confusion Matrix`;
  return c;
});

predictButton.$click.subscribe(async () => {
  await Promise.all(batchPredictions.map((x) => x.clear()));
  await Promise.all(batchPredictions.map((x, i) => x.predict(classifiers[i], trainingSet, 'data')));
});

// -----------------------------------------------------------
// DASHBOARDS
// -----------------------------------------------------------

const dash = dashboard({
  title: 'Marcelle: Skin Lesion Classification',
  author: 'Marcelle Pirates Crew',
});
$dashboardPage = dash.$page;

dash
  .page('Model Selector')
  .use(runSelectors, modelSelectors, modelLoaders, classifiers)
  .use(lossPlots, accuracyPlots);
// .use(trainingPlots, runSummary, modelSummary);
dash.page('Real-time Testing').useLeft(source, sourceImages).use(predictionPlots);
dash
  .page('Batch Testing')
  .useLeft(source, label)
  .use(trainingSetBrowser, predictButton, confusionMatrices);

dash.settings
  .dataStores(store)
  .datasets(trainingSet)
  .models(...classifiers);

dash.start();
