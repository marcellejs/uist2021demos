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
  parameters,
  classificationPlot,
  notification,
  trainingProgress,
  sketchpad,
  textfield,
  trainingPlot,
} from '@marcellejs/core';

// -----------------------------------------------------------
// INPUT PIPELINE & DATA CAPTURE
// -----------------------------------------------------------

const input = sketchpad();
const featureExtractor = mobilenet();

const store = dataStore({ location: 'localStorage' });
const trainingSet = dataset({ name: 'TrainingSet-sketch', dataStore: store });

const labelField = textfield();
labelField.title = 'Correct the prediction if necessary';
labelField.$text.set('...');
const addToDataset = button({ text: 'Add to Dataset and Train' });
addToDataset.title = 'Improve the classifier';

const $instances = addToDataset.$click
  .sample(input.$images.zip((thumbnail, data) => ({ thumbnail, data }), input.$thumbnails))
  .map(async (instance) => ({
    ...instance,
    type: 'sketch',
    label: labelField.$text.value,
    features: await featureExtractor.process(instance.data),
  }))
  .awaitPromises();

trainingSet.capture($instances);

const trainingSetBrowser = datasetBrowser(trainingSet);

// -----------------------------------------------------------
// TRAINING
// -----------------------------------------------------------

const b = button({ text: 'Train' });
const classifier = mlp({ layers: [64, 32], epochs: 20, dataStore: store });
classifier.sync('sketch-classifier');

b.$click.subscribe(() => classifier.train(trainingSet));

trainingSet.$changes.subscribe((changes) => {
  for (let i = 0; i < changes.length; i++) {
    if (changes[i].level === 'instance' && changes[i].type === 'created') {
      if (
        Object.values(trainingSet.$classes.value)
          .map((x) => x.length)
          .reduce((x, y) => x || y < 2, false)
      ) {
        notification({
          title: 'Tip',
          message: 'You need to record at least two examples per class',
          duration: 5000,
        });
      } else if (trainingSet.$labels.value.length < 2) {
        notification({
          title: 'Tip',
          message: 'You need to have at least two classes to train the model',
          duration: 5000,
        });
      } else {
        classifier.train(trainingSet);
      }
      break;
    } else if (changes[i].level === 'class' && ['deleted', 'renamed'].includes(changes[i].type)) {
      classifier.train(trainingSet);
    }
  }
});

const params = parameters(classifier);
const prog = trainingProgress(classifier);
const plotTraining = trainingPlot(classifier);

// -----------------------------------------------------------
// REAL-TIME PREDICTION
// -----------------------------------------------------------

const $features = input.$images.map((img) => featureExtractor.process(img)).awaitPromises();
const $trainingSuccess = classifier.$training.filter((x) => x.status === 'success');
const $predictions = $features
  .merge($trainingSuccess.sample($features))
  .map((features) => classifier.predict(features))
  .awaitPromises()
  .filter((x) => !!x);

const plotResults = classificationPlot($predictions);

$predictions.subscribe(({ label }) => {
  labelField.$text.set(label);
});

// const $instances = addToDataset.$click
//   .sample(input.$images)
//   .map(async (img) => ({
//     type: 'sketch',
//     data: img,
//     label: labelField.$text.value,
//     features: await featureExtractor.process(instance.data),
//   }))
//   .awaitPromises();
//
// trainingSet.capture($instances);

// -----------------------------------------------------------
// DASHBOARDS
// -----------------------------------------------------------

const dash = dashboard({
  title: 'Marcelle Example - Dashboard',
  author: 'Marcelle Pirates Crew',
});

dash
  .page('Online Learning')
  .useLeft(input, featureExtractor)
  .use(plotResults, [labelField, addToDataset], prog, trainingSetBrowser);
dash.page('Offline Training').useLeft(trainingSetBrowser).use(params, b, prog, plotTraining);
dash.settings.dataStores(store).datasets(trainingSet).models(classifier, featureExtractor);

dash.start();

// -----------------------------------------------------------
// HELP MESSAGES
// -----------------------------------------------------------

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
