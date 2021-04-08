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
  text,
  trainingPlot,
  notification,
} from '@marcellejs/core';
import { pythonTrained } from './modules';
import { classifier, instances, labels, source, sourceImages, store } from './common';

let $dashboardPage;

// -----------------------------------------------------------
// PYTHON-TRAINED MODULE
// -----------------------------------------------------------

const model2 = pythonTrained({ dataStore: store });

const selectRun = select();
selectRun.title = 'Select Training Run:';
model2.$runs.subscribe((runs) => {
  selectRun.$options.set(runs);
  if (!selectRun.$value.value) {
    selectRun.$value.set(runs[0]);
  }
});
selectRun.$value.subscribe(model2.loadRun.bind(model2));

const selectModel = select();
selectModel.title = 'Select Model Version:';
model2.$checkpoints.subscribe((checkpoint) => {
  selectModel.$options.set(
    checkpoint.map((x) => (x.epoch === 'final' ? 'Final' : `Epoch: ${x.epoch}`)),
  );
});

const loadModelBtn = button({ text: 'Load Model ' });
loadModelBtn.title = 'Load Model Version';
loadModelBtn.$click.sample(selectModel.$value).subscribe((checkpoint) => {
  if (checkpoint) {
    const epoch = checkpoint === 'Final' ? 'final' : checkpoint.split('Epoch: ')[1];
    const modelPath = model2.$checkpoints.value.filter((x) => x.epoch.toString() === epoch)[0].url;
    const url = `${store.location}/tfjs-models/${modelPath}`;
    // eslint-disable-next-line no-console
    console.log('Loaded model:', url);
    classifier.loadFromUrl(url);
  }
});

const runSummary = text({ text: 'Waiting for run data...' });
runSummary.title = 'Run Information';
model2.$run
  .filter((x) => !!Object.keys(x).length)
  .subscribe((x) => {
    let summaryText = '<div class="min-w-full">';
    summaryText += `<h2>Training Status: ${x.status}`;
    if (!['idle', 'start'].includes(x.status)) {
      summaryText += ` (epoch ${x.epoch + 1}/${x.epochs})`;
    }
    summaryText += '</h2>';
    summaryText += `<ul class="list-disc list-inside ml-6">`;
    summaryText += `<li>Start date: ${x.run_start_at}</li>`;
    summaryText += `<li>Source: ${x.source}</li>`;
    summaryText += `<li>Logged Values: ${Object.keys(x.logs).join(', ')}</li>`;
    summaryText += `<li>Checkpoints: ${x.checkpoints.length}</li>`;
    summaryText += `</ul>`;
    summaryText += `<h2>Parameters:</h2>`;
    summaryText += `<table class="table-auto mx-6"><thead><tr>
      <th class="bg-gray-200 text-gray-600 border-2 border-gray-200">Parameter</th>
      <th class="bg-gray-200 text-gray-600 border-2 border-gray-200">Value</th>
      </tr></thead><tbody class="bg-gray-200">`;
    Object.entries(x.params).forEach(([k, v]) => {
      summaryText += `<tr class="bg-white border-2 border-gray-200">
        <td class="px-16 py-1">${k}</td>
        <td class="px-16 py-1">${Array.isArray(v) ? `[${v.join(', ')}]` : v}</td>
      </tr>`;
    });
    summaryText += '</tbody></table>';
    runSummary.$text.set(summaryText);
  });

const modelSummary = text({ text: 'Waiting for a model...' });
modelSummary.title = 'Model Summary';
model2.$run
  .filter((x) => !!Object.keys(x).length)
  .subscribe((x) => {
    let s = '<div style="margin: auto; font-size: 0.8em;"><pre>';
    s += `${x.model.summary}<br>`;
    s += '</pre></div>';
    modelSummary.$text.set(s);
  });

const plotTraining = trainingPlot(model2, {
  loss: ['loss', 'val_loss'],
  accuracy: ['accuracy', 'val_accuracy'],
});

// -----------------------------------------------------------
// REAL-TIME PREDICTION
// -----------------------------------------------------------

const predictionStream = source.$images
  .filter(() => $dashboardPage.value === 'real-time-testing')
  .map(async (img) => classifier.predict(img))
  .awaitPromises();

const plotResults = classificationPlot(predictionStream);

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

const batchTesting = batchPrediction({ name: 'ml-dashboard', dataStore: store });
const predictButton = button({ text: 'Update predictions' });
predictButton.title = 'Batch predictions';
const confMat = confusionMatrix(batchTesting);
// confMat.title = 'Confusion Matrix';

predictButton.$click.subscribe(async () => {
  await batchTesting.clear();
  await batchTesting.predict(classifier, trainingSet, 'data');
});

const selectClinicianModel = button({ text: 'Share' });
selectClinicianModel.title = 'Share the selected model with the clinician';
selectClinicianModel.$click.subscribe(async () => {
  await classifier.save(true, { name: 'clinician-model' });
  notification({
    title: 'Model Synchronized',
    message: 'The model was saved for the clinician',
    duration: 5000,
  });
});

// -----------------------------------------------------------
// CLINICIAN'S Data
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

const dash = dashboard({
  title: 'Marcelle: Skin Lesion Classification',
  author: 'Marcelle Pirates Crew',
});
$dashboardPage = dash.$page;

dash
  .page('Model Selector')
  .useLeft(selectRun, selectModel, loadModelBtn, classifier)
  .use(plotTraining, runSummary, modelSummary);
dash.page('Real-time Testing').useLeft(classifier, source).use([sourceImages, plotResults]);
dash.page('Batch Testing').useLeft(source, label).use(trainingSetBrowser, predictButton, confMat);
dash
  .page('Collaboration')
  .use('Share Model', selectClinicianModel)
  .use("Clinician's data", [correctSetBrowser, incorrectSetBrowser]);

dash.settings.dataStores(store).datasets(trainingSet).models(classifier);

dash.start();
