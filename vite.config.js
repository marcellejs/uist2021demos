/* eslint-disable import/no-extraneous-dependencies */
import svelte from 'rollup-plugin-svelte';
import { resolve } from 'path';
import { render } from 'ejs';
import examples from './meta.json';

function injectHtml({ injectData = {}, injectOptions = {}, tags = [] } = {}) {
  return {
    name: 'vite:injectHtml',
    transformIndexHtml: {
      enforce: 'pre',
      transform(html) {
        return {
          html: render(html, injectData, injectOptions),
          tags,
        };
      },
    },
  };
}

const toCamelCase = (str) => {
  const s =
    str &&
    str
      .match(/[A-Z]{2,}(?=[A-Z][a-z]+[0-9]*|\b)|[A-Z]?[a-z]+[0-9]*|[A-Z]|[0-9]+/g)
      .map((x) => x.slice(0, 1).toUpperCase() + x.slice(1).toLowerCase())
      .join('');
  return s.slice(0, 1).toLowerCase() + s.slice(1);
};

export default {
  plugins: [
    injectHtml({
      injectData: {
        examples,
      },
    }),
    svelte(),
  ],
  build: {
    rollupOptions: {
      input: examples.reduce(
        (o, x) => ({ ...o, [toCamelCase(x.path)]: resolve(__dirname, x.path, 'index.html') }),
        {
          main: resolve(__dirname, 'index.html'),
        },
      ),
    },
  },
  optimizeDeps: {
    exclude: ['@marcellejs/backend'],
  },
};
