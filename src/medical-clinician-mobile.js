import '@marcellejs/core/dist/marcelle.css';
import { datasetBrowser, dashboard, dataset, webcam, button } from '@marcellejs/core';
import { store } from './common';

const mobileDataset = dataset({ name: 'mobile', dataStore: store });

const mobileDatasetBrowser = datasetBrowser(mobileDataset);
mobileDatasetBrowser.title = 'Dataset: Captured from mobile phone';

const input = webcam({ width: 224, height: 224 });

const capture = button({ text: 'Capture' });
capture.title = 'Capture images';

const $instances = capture.$click.sample(
  input.$images.zip(
    (thumbnail, data) => ({ thumbnail, data, type: 'image', label: 'unlabeled' }),
    input.$thumbnails,
  ),
);

mobileDataset.capture($instances);

const dash = dashboard({
  title: 'Marcelle: Skin Lesion Classification',
  author: 'Louise',
});

dash.page('Main').use(input, capture, mobileDatasetBrowser);

dash.settings.dataStores(store).datasets(mobileDatasetBrowser);

dash.start();
