import '@marcellejs/core/dist/marcelle.css';
import { dataset, datasetBrowser, dataStore, imageUpload, tfGenericModel } from '@marcellejs/core';
import { InstanceViewer } from './instance-viewer';

const location = 'https://marcelle-uist2021.herokuapp.com';
// const location = 'http://localhost:3030';
export const store = dataStore({ location });

// -----------------------------------------------------------
// INPUT PIPELINE & CLASSIFICATION
// -----------------------------------------------------------

const mobileDataset = dataset({ name: 'mobile', dataStore: store });

export const mobileDatasetBrowser = datasetBrowser(mobileDataset);
mobileDatasetBrowser.title = 'Dataset: Captured from mobile phone';

const $mobileInstances = mobileDatasetBrowser.$selected
  .filter((x) => x.length === 1)
  .map(([id]) => mobileDataset.getInstance(id))
  .awaitPromises()
  .map((x) => {
    const { id, datasetName, ...rest } = x;
    return rest;
  });

const $selectedMobileImage = $mobileInstances.map(({ data }) => data);

export const source = imageUpload();

export const $inputImages = source.$images.merge($selectedMobileImage);

export const sourceImages = new InstanceViewer($inputImages);

export const $uploadInstances = source.$images.zip(
  (thumbnail, data) => ({ thumbnail, data, type: 'image', label: 'unlabeled' }),
  source.$thumbnails,
);
export const instances = $uploadInstances.merge($mobileInstances);

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
