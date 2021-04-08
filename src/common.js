import '@marcellejs/core/dist/marcelle.css';
import { dataStore, imageUpload, tfGenericModel } from '@marcellejs/core';
import { InstanceViewer } from './instance-viewer';

const location = 'http://localhost:3030';
export const store = dataStore({ location });

// -----------------------------------------------------------
// INPUT PIPELINE & CLASSIFICATION
// -----------------------------------------------------------

export const source = imageUpload();
export const sourceImages = new InstanceViewer(source.$images);
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
}).sync('skin-lesion-classifier');

classifier.labels = labels;
