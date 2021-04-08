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
} from '@marcellejs/core';
import {
  classifier,
  instances,
  labels,
  mobileDatasetBrowser,
  source,
  sourceImages,
  store,
} from './common';

store.connect().then(() => {
  classifier.sync('clinician-model');
});

// -----------------------------------------------------------
// CAPTURE TO DATASET
// -----------------------------------------------------------

const correctSet = dataset({ name: 'CorrectSet', dataStore: store });
const incorrectSet = dataset({ name: 'IncorrectSet', dataStore: store });

const correctSetBrowser = datasetBrowser(correctSet);
correctSetBrowser.title = 'Dataset: Correct Predictions';
const incorrectSetBrowser = datasetBrowser(incorrectSet);
incorrectSetBrowser.title = 'Dataset: Incorrect Predictions';

// -----------------------------------------------------------
// BATCH PREDICTION
// -----------------------------------------------------------

const batchCorrect = batchPrediction({ name: 'correct', dataStore: store });
const batchIncorrect = batchPrediction({ name: 'incorrect', dataStore: store });
const predictButton = button({ text: 'Update confusion matrices' });
predictButton.title = '';
const confusionMatrixCorrect = confusionMatrix(batchCorrect);
const confusionMatrixIncorrect = confusionMatrix(batchIncorrect);
confusionMatrixCorrect.title = 'Confusion Matrix (Correct)';
confusionMatrixIncorrect.title = 'Confusion Matrix (Incorrect)';

predictButton.$click.subscribe(async () => {
  await batchCorrect.clear();
  await batchIncorrect.clear();
  await batchCorrect.predict(classifier, correctSet, 'data');
  await batchIncorrect.predict(classifier, incorrectSet, 'data');
});

// -----------------------------------------------------------
// REAL-TIME PREDICTION
// -----------------------------------------------------------

const predictionStream = source.$images.map(async (img) => classifier.predict(img)).awaitPromises();

const plotResults = classificationPlot(predictionStream);

const label = select({ options: labels });
label.title = 'Correct the prediction if necessary';

let predictedLabel;
predictionStream.subscribe(({ label: l }) => {
  predictedLabel = l;
  label.$value.set(l);
});

const addToDataset = button({ text: 'Confirm diagnosis' });
addToDataset.title = 'Add to the dataset';

addToDataset.$click
  .snapshot((instance) => ({ ...instance, label: label.$value.value }), instances)
  .subscribe((instance) => {
    if (predictedLabel === instance.label) {
      correctSet.addInstance(instance);
    } else {
      incorrectSet.addInstance(instance);
    }
  });

const quality = text({ text: 'Waiting for predictions...' });
correctSet.$count
  .combine((a, b) => (100 * b) / (a + b || 1), incorrectSet.$count)
  .subscribe((percent) => {
    quality.$text.set(
      `You evaluated ${percent.toFixed(0)}% of ${
        correctSet.$count.value + incorrectSet.$count.value
      } tested images as correct.`,
    );
  });

// -----------------------------------------------------------
// DASHBOARDS
// -----------------------------------------------------------

const dash = dashboard({
  title: 'Marcelle: Skin Lesion Classification',
  author: 'Marcelle Pirates Crew',
});

dash
  .page('Check Images')
  .useLeft(source, classifier, mobileDatasetBrowser)
  .use([sourceImages, plotResults], [label, addToDataset], quality);

dash
  .page('Inspect Misclassifications')
  .use([correctSetBrowser, incorrectSetBrowser], predictButton, [
    confusionMatrixCorrect,
    confusionMatrixIncorrect,
  ]);

dash.settings.dataStores(store).datasets(correctSetBrowser, incorrectSetBrowser).models(classifier);

dash.start();
