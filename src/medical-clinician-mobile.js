import '@marcellejs/core/dist/marcelle.css';
import { datasetBrowser, dashboard, dataset, webcam, button, text } from '@marcellejs/core';
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

const disclaimer = text({
  text:
    '<span style="color: red; ">Webcam image capture still suffers bugs. The captured image might be frozen from the webcam startup.</span>',
});
disclaimer.title = 'Disclaimer';

const dash = dashboard({
  title: 'Marcelle: Skin Lesion Classification',
  author: 'Louise',
});

dash.page('Main').use(disclaimer, input, capture, mobileDatasetBrowser);

dash.settings.dataStores(store).datasets(mobileDatasetBrowser);

dash.start();
