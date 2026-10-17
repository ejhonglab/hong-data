// Web worker for ClusterPool (cluster.js). Messages in:
//   {type: 'dist', job, r0, r1, [vectors, n, d, metric]}  distance block of rows r0..r1-1
//       (the vectors come with a job's first block for this worker, and are kept)
//   {type: 'forget', job}                                  drop a job's vectors
//   {type: 'link', job, n, fill, blocks: [{r0, r1, data}]}  average linkage on all blocks
// out: {type: 'block', job, r0, r1, block, maxD}, {type: 'progress', job, fraction},
//      {type: 'result', job, order, merges}, {type: 'error', job, message, outOfMemory}
import { averageLinkage, leafOrder, prepare, rowDistances } from './cluster.js';

const prepared = new Map();

self.onmessage = ({ data: m }) => {
  try {
    if (m.type === 'dist') {
      if (m.vectors) prepared.set(m.job, prepare(m.vectors, m.n, m.d, m.metric));
      const { block, maxD } = rowDistances(prepared.get(m.job), m.r0, m.r1);
      self.postMessage({ type: 'block', job: m.job, r0: m.r0, r1: m.r1, block, maxD }, [block.buffer]);
    } else if (m.type === 'forget') {
      prepared.delete(m.job);
    } else if (m.type === 'link') {
      const merges = averageLinkage(m.blocks, m.n, m.fill, (fraction) => self.postMessage({ type: 'progress', job: m.job, fraction }));
      const order = leafOrder(merges, m.n);
      self.postMessage({ type: 'result', job: m.job, order, merges }, [order.buffer, merges.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', job: m.job, message: String((err && err.message) || err), outOfMemory: err instanceof RangeError });
  }
};
