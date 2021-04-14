import { RunManager } from './run-manager.module';

export function runManager(options) {
  return new RunManager(options);
}
