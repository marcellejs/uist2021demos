import { logger, Module, Stream } from '@marcellejs/core';

export class PythonTrained extends Module {
  constructor({ dataStore }) {
    super();
    this.title = 'Python-trained model';
    this.dataStore = dataStore;
    this.$run = new Stream({}, true);
    this.$runs = new Stream([], true);
    this.$training = new Stream({ status: 'idle' }, true);
    this.$checkpoints = new Stream([], true);
    this.start();
    this.dataStore
      .connect()
      .then(() => {
        this.setup();
      })
      .catch(() => {
        logger.log('[dataset] dataStore connection failed');
      });
  }

  async setup() {
    this.runService = this.dataStore.service('runs');
    const { data } = await this.runService.find();
    this.$runs.set(data.map((x) => x.run_start_at));
  }

  async loadRun(run) {
    if (!run || !this.runService) return;
    const { data } = await this.runService.find({
      query: { run_start_at: run, $limit: 1, $sort: { updatedAt: -1 } },
    });
    this.$training.set({
      status: data[0].status,
      epoch: data[0].epoch,
      epochs: data[0].epochs,
      data: data[0].logs,
    });
    this.$run.set(data[0]);
    this.$checkpoints.set(data[0].checkpoints);
  }

  // eslint-disable-next-line class-methods-use-this
  mount() {}
}
