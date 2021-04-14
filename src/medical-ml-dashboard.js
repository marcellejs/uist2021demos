import '@marcellejs/core/dist/marcelle.css';
import {
  datasetBrowser,
  button,
  dashboard,
  dataset,
  classificationPlot,
  trainingPlot,
  notification,
} from '@marcellejs/core';
import { runManager } from './modules';
import { $inputImages, instances, source, sourceImages, store } from './common';
import {
  createModel,
  createPredictionStream,
  managerControls,
  modelSummary,
  runSummary,
  setupBatchPrediction,
  setupTestSet,
} from './ml-utils';

const dash = dashboard({
  title: 'Marcelle: Skin Lesion Classification',
  author: 'Louise',
});

// -----------------------------------------------------------
// Training Experiment Management
// -----------------------------------------------------------

const classifier = createModel();
const manager = runManager({ dataStore: store });

const { selectRun, selectModel, loadModel } = managerControls(manager, classifier);
const runInfo = runSummary(manager);
const modelInfo = modelSummary(manager);

const plotTraining = trainingPlot(manager, {
  loss: ['loss', 'val_loss'],
  accuracy: ['accuracy', 'val_accuracy'],
});

// -----------------------------------------------------------
// REAL-TIME PREDICTION
// -----------------------------------------------------------

const predictionStream = createPredictionStream(
  $inputImages.filter(() => dash.$page.value === 'real-time-testing'),
  classifier,
);

const plotResults = classificationPlot(predictionStream);

// -----------------------------------------------------------
// BATCH PREDICTION
// -----------------------------------------------------------

const { testSet, selectLabel } = setupTestSet(
  instances.filter(() => dash.$page.value === 'batch-testing'),
);
const testSetBrowser = datasetBrowser(testSet);

const predictButton = button({ text: 'Update predictions' });
predictButton.title = 'Batch predictions';
const { confMat } = setupBatchPrediction(testSet, classifier, predictButton);

// -----------------------------------------------------------
// SHARE MODEL WITH THE CLINICIAN
// -----------------------------------------------------------

const selectClinicianModel = button({ text: 'Share' });
selectClinicianModel.title = 'Share the selected model with the clinician';
selectClinicianModel.$click.subscribe(async () => {
  const { data } = await classifier.service.find({
    query: {
      name: 'clinician-model',
      $select: ['id'],
      $limit: 1,
      $sort: {
        updatedAt: -1,
      },
    },
  });
  if (data.length === 1) {
    await classifier.save('clinician-model', {}, data[0].id);
  } else {
    await classifier.save('clinician-model');
  }
  notification({
    title: 'Model Synchronized',
    message: 'The model was saved for the clinician',
    duration: 5000,
  });
});

// -----------------------------------------------------------
// CLINICIAN'S DATA
// -----------------------------------------------------------

const correctSet = dataset({ name: 'CorrectSet', dataStore: store });
const incorrectSet = dataset({ name: 'IncorrectSet', dataStore: store });

const correctSetBrowser = datasetBrowser(correctSet);
correctSetBrowser.title = 'Correct Predictions';
const incorrectSetBrowser = datasetBrowser(incorrectSet);
incorrectSetBrowser.title = 'Incorrect Predictions';

// -----------------------------------------------------------
// DASHBOARDS
// -----------------------------------------------------------

dash
  .page('Model Selector')
  .useLeft(selectRun, selectModel, loadModel, classifier)
  .use(plotTraining, runInfo, modelInfo);
dash.page('Real-time Testing').useLeft(classifier, source).use([sourceImages, plotResults]);
dash.page('Batch Testing').useLeft(source, selectLabel).use(testSetBrowser, predictButton, confMat);
dash
  .page('Collaboration')
  .use('Share Model', selectClinicianModel)
  .use("Clinician's data", [correctSetBrowser, incorrectSetBrowser]);

dash.settings.dataStores(store).datasets(testSet, correctSet, incorrectSet).models(classifier);

dash.start();
