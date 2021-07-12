import '@marcellejs/core/dist/marcelle.css';
import {
  batchPrediction,
  datasetBrowser,
  button,
  confusionMatrix,
  dashboard,
  dataset,
  confidencePlot,
  select,
  text,
  imageDisplay,
} from '@marcellejs/core';
import {
  $inputImages,
  instances,
  labels,
  // mobileDatasetBrowser,
  source,
  sourceImages,
  store,
} from './common';
import { createModel, createPredictionStream } from './ml-utils';

const classifier = createModel();
classifier.sync('clinician-model');

// -----------------------------------------------------------
// DATASET DEFINITIONS
// -----------------------------------------------------------

const correctSet = dataset('CorrectSet', store);
const incorrectSet = dataset('IncorrectSet', store);

const correctSetBrowser = datasetBrowser(correctSet);
correctSetBrowser.title = 'Dataset: Correct Predictions';
const incorrectSetBrowser = datasetBrowser(incorrectSet);
incorrectSetBrowser.title = 'Dataset: Incorrect Predictions';

// -----------------------------------------------------------
// REAL-TIME PREDICTION
// -----------------------------------------------------------

const predictionStream = $inputImages.map(async (img) => classifier.predict(img)).awaitPromises();

const plotResults = confidencePlot(predictionStream);

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
  .snapshot((instance) => ({ ...instance, y: label.$value.value }), instances)
  .subscribe((instance) => {
    if (predictedLabel === instance.y) {
      correctSet.create(instance);
    } else {
      incorrectSet.create(instance);
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
// INSPECT MISCLASSIFICATIONS
// -----------------------------------------------------------

const batchIncorrect = batchPrediction({ name: 'incorrect' });
const predictButton = button({ text: 'Update Confusion Matrix' });
predictButton.title = '';
const confusionMatrixIncorrect = confusionMatrix(batchIncorrect);
confusionMatrixIncorrect.title = 'Confusion Matrix (Incorrect)';

predictButton.$click.subscribe(async () => {
  await batchIncorrect.clear();
  await batchIncorrect.predict(classifier, incorrectSet);
});

function selectedStream(ds, dsBrowser) {
  return dsBrowser.$selected
    .filter((x) => x.length === 1)
    .map(([id]) => ds.get(id, ['x']))
    .awaitPromises()
    .map(({ x }) => x);
}

const $selectedImage = selectedStream(correctSet, correctSetBrowser).merge(
  selectedStream(incorrectSet, incorrectSetBrowser),
);
const $predictions2 = createPredictionStream($selectedImage, classifier);
const plotResults2 = confidencePlot($predictions2);

const helpText = text({ text: "Select instances to review the model's predictions" });
helpText.title = 'hint';

// -----------------------------------------------------------
// DASHBOARDS
// -----------------------------------------------------------

const dash = dashboard({
  title: 'Marcelle: Skin Lesion Classification',
  author: 'Louise',
});

dash
  .page('Check Images')
  .sidebar(source, classifier) // , mobileDatasetBrowser
  .use([sourceImages, plotResults], quality, label, addToDataset);

dash
  .page('Inspect Misclassifications')
  .sidebar(classifier, predictButton, confusionMatrixIncorrect)
  .use(helpText)
  .use([correctSetBrowser, incorrectSetBrowser], [imageDisplay($selectedImage), plotResults2]);

dash.settings.dataStores(store).datasets(correctSet, incorrectSet).models(classifier);

dash.show();
