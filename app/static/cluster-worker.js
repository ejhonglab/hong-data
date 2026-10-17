// Web worker: clustering off the main thread (see cluster.js).
// message in: { jobs: [{ data: Float32Array, n, d, metric }] }
// message out: { results: [{ order, merges }] } or { error }
import { cluster } from './cluster.js';

self.onmessage = (e) => {
  try {
    const results = e.data.jobs.map((j) => cluster(j.data, j.n, j.d, j.metric));
    self.postMessage({ results }, results.flatMap((r) => [r.order.buffer, r.merges.buffer]));
  } catch (err) {
    self.postMessage({ error: String((err && err.message) || err) });
  }
};
