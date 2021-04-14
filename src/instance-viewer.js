import '@marcellejs/core/dist/marcelle.css';
import { Module } from '@marcellejs/core';

export class InstanceViewer extends Module {
  constructor(imgStream) {
    super();
    this.title = 'instance viewer';
    this.imgStream = imgStream;
  }

  mount(target) {
    this.target = target || document.querySelector(`#${this.id}`);
    this.canvas = document.createElement('canvas');
    this.canvas.classList.add('w-full', 'max-w-full');
    const ctx = this.canvas.getContext('2d');
    this.target.appendChild(this.canvas);
    this.unSub = this.imgStream.subscribe((img) => {
      this.canvas.width = img.width;
      this.canvas.height = img.height;
      ctx.putImageData(img, 0, 0);
    });
  }

  destroy() {
    if (this.target && this.canvas) {
      this.target.removeChild(this.canvas);
    }
    this.unSub();
  }

  // eslint-disable-next-line class-methods-use-this
  unSub() {}
}
