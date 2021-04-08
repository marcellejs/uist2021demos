import '@marcellejs/core/dist/marcelle.css';
import { datasetBrowser, dashboard, dataset, webcam, button } from '@marcellejs/core';
import { classifier, store } from './common';

const mobileDataset = dataset({ name: 'mobile', dataStore: store });

const mobileDatasetBrowser = datasetBrowser(mobileDataset);
mobileDatasetBrowser.title = 'Dataset: Captured from mobile phone';

const input = webcam({ width: 224, height: 224 });

const capture = button({ text: 'Capture' });
capture.title = 'Capture images';

const $instances = capture.$click.sample(
  input.$thumbnails.map((thumbnail) => ({
    type: 'image',
    data: input.$images.value,
    label: 'unlabeled',
    thumbnail,
  })),
);

mobileDataset.capture($instances);

const dash = dashboard({
  title: 'Marcelle: Skin Lesion Classification',
  author: 'Marcelle Pirates Crew',
});

dash.page('Main').use(input, capture, mobileDatasetBrowser);

dash.settings.dataStores(store).datasets(mobileDatasetBrowser).models(classifier);

dash.start();
