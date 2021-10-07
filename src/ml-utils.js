import {
  batchPrediction,
  button,
  confusionMatrix,
  dataset,
  notification,
  select,
  text,
  tfjsModel,
} from '@marcellejs/core';
import { labels, store } from './common';

export function managerControls(history, classifier, action = 'select model') {
  const selectCheckpoint = select();
  selectCheckpoint.title = 'Select Model Checkpoint:';
  let crtRun;
  history.$actions.subscribe(({ name, data }) => {
    if (name === action) {
      crtRun = data;
      selectCheckpoint.$options.set(data.checkpoints.map((x) => x.name));
      if (!selectCheckpoint.$value.value) {
        selectCheckpoint.$value.set(selectCheckpoint.$options.value[0]);
      }
    }
  });

  const loadModel = button({ text: 'Load Model ' });
  loadModel.title = 'Load Model Version';
  loadModel.$click.sample(selectCheckpoint.$value).subscribe((checkpointName) => {
    if (checkpointName) {
      const checkpoint = crtRun.checkpoints.filter((x) => x.name === checkpointName)[0];
      console.log('checkpoint', checkpoint);
      const url = `${store.location}/${checkpoint.service}/${checkpoint.id}/model.json`;
      console.log('url', url);
      // eslint-disable-next-line no-console
      console.log('Loading model:', url);
      classifier.loadFromUrl(url);
    }
  });

  return { selectCheckpoint, loadModel };
}

export function modelSummary(manager) {
  const summary = text({ text: 'Waiting for a model...' });
  summary.title = 'Model Summary';
  manager.$run
    .filter((x) => !!Object.keys(x).length)
    .subscribe((x) => {
      let s = '<div style="margin: auto; font-size: 0.8em;"><pre>';
      s += `${x.model.summary}<br>`;
      s += '</pre></div>';
      summary.$text.set(s);
    });
  return summary;
}

export function createModel() {
  const classifier = tfjsModel({
    inputType: 'image',
    taskType: 'classification',
    dataStore: store,
  });
  classifier.labels = labels;
  return classifier;
}

export function createPredictionStream($inputImages, classifier) {
  return $inputImages
    .tap(() => {
      if (!classifier.ready) {
        notification({
          title: 'No model loaded',
          message: 'Select a model a load it on the first page',
        });
      }
    })
    .filter(() => classifier.ready)
    .map(async (img) => classifier.predict(img))
    .awaitPromises();
}

export function setupTestSet(instances) {
  const testSet = dataset('test-set', store);

  const selectLabel = select({ options: labels });
  selectLabel.title = 'Select label:';
  instances
    .map((x) => ({ ...x, y: selectLabel.$value.value }))
    .subscribe(testSet.create.bind(testSet));
  return { testSet, selectLabel };
}

let nBatchPred = 0;
export function setupBatchPrediction(testSet, classifier, predictButton) {
  const batchTesting = batchPrediction({ name: `medical-preds-${nBatchPred++}` });
  const confMat = confusionMatrix(batchTesting);

  predictButton.$click.subscribe(async () => {
    if (!classifier.ready) {
      notification({
        title: 'No model loaded',
        message: 'Select a model a load it on the first page',
      });
      return;
    }
    await batchTesting.clear();
    await batchTesting.predict(classifier, testSet, 'data');
  });
  return { confMat };
}
