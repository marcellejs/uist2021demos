import {
  batchPrediction,
  button,
  confusionMatrix,
  dataset,
  notification,
  select,
  text,
  tfGenericModel,
} from '@marcellejs/core';
import { labels, store } from './common';

export function managerControls(manager, classifier) {
  const selectRun = select();
  selectRun.title = 'Select Training Run:';
  manager.$runs.subscribe((runs) => {
    selectRun.$options.set(runs);
    if (!selectRun.$value.value) {
      selectRun.$value.set(runs[0]);
    }
  });
  selectRun.$value.subscribe(manager.loadRun.bind(manager));

  const selectModel = select();
  selectModel.title = 'Select Model Version:';
  manager.$checkpoints.subscribe((checkpoint) => {
    selectModel.$options.set(
      checkpoint.map((x) => (x.epoch === 'final' ? 'Final' : `Epoch: ${x.epoch}`)),
    );
    if (!selectModel.$value.value) {
      selectModel.$value.set(selectModel.$options.value[0]);
    }
  });

  const loadModel = button({ text: 'Load Model ' });
  loadModel.title = 'Load Model Version';
  loadModel.$click.sample(selectModel.$value).subscribe((checkpoint) => {
    if (checkpoint) {
      const epoch = checkpoint === 'Final' ? 'final' : checkpoint.split('Epoch: ')[1];
      // eslint-disable-next-line no-underscore-dangle
      const modelId = manager.$checkpoints.value.filter((x) => x.epoch.toString() === epoch)[0]._id;
      const url = `${manager.dataStore.location}/tfjs-models/${modelId}/model.json`;
      // eslint-disable-next-line no-console
      console.log('Loading model:', url);
      classifier.loadFromUrl(url);
    }
  });

  return { selectRun, selectModel, loadModel };
}

export function runSummary(manager) {
  const summary = text({ text: 'Waiting for run data...' });
  summary.title = 'Run Information';
  manager.$run
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
      summary.$text.set(summaryText);
    });
  return summary;
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
  const classifier = tfGenericModel({
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
  const testSet = dataset({ name: 'test-set', dataStore: store });

  const selectLabel = select({ options: labels });
  selectLabel.title = 'Select label:';
  testSet.capture(instances.map((y) => ({ ...y, label: selectLabel.$value.value })));
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
    console.log('batchTesting', batchTesting);
  });
  return { confMat };
}
