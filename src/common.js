import '@marcellejs/core/dist/marcelle.css';
import { dataset, datasetBrowser, dataStore, imageUpload, tfGenericModel } from '@marcellejs/core';
import { InstanceViewer } from './instance-viewer';

const location = 'https://marcelle-uist2021-test.herokuapp.com';
export const store = dataStore({ location });

// -----------------------------------------------------------
// INPUT PIPELINE & CLASSIFICATION
// -----------------------------------------------------------

const mobileDataset = dataset({ name: 'mobile', dataStore: store });

export const mobileDatasetBrowser = datasetBrowser(mobileDataset);
mobileDatasetBrowser.title = 'Dataset: Captured from mobile phone';

const $selectedMobileImage = mobileDatasetBrowser.$selected
  .filter((x) => x.length === 1)
  .map(([id]) => mobileDataset.getInstance(id, ['data']))
  .awaitPromises()
  .map(({ data }) => data);
export const source = imageUpload();

export const sourceImages = new InstanceViewer(source.$images.merge($selectedMobileImage));
export const instances = source.$thumbnails.map((thumbnail) => ({
  type: 'image',
  data: source.$images.value,
  label: 'unlabeled',
  thumbnail,
}));

export const labels = [
  'Actinic keratosis',
  'Basal cell carcinoma',
  'Benign keratosis',
  'Dermatofibroma',
  'Melanoma',
  'Melanocytic nevus',
  'Vascular lesion',
];

export const classifier = tfGenericModel({
  inputType: 'image',
  taskType: 'classification',
  dataStore: store,
});

classifier.labels = labels;
