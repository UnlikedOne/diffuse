import logging
import queue
import threading

log = logging.getLogger("diffuse.worker.batching")


class Job:
    __slots__ = (
        "tensor_in",
        "is_input_ids",
        "session_id",
        "use_cache",
        "top_k",
        "done",
        "result",
        "error",
    )

    def __init__(self, tensor_in, is_input_ids, session_id, use_cache, top_k):
        self.tensor_in = tensor_in
        self.is_input_ids = is_input_ids
        self.session_id = session_id
        self.use_cache = use_cache
        self.top_k = top_k
        self.done = threading.Event()
        self.result = None
        self.error = None

    def batchable(self) -> bool:
        return (
            self.use_cache
            and bool(self.session_id)
            and self.tensor_in.dim() >= 2
            and self.tensor_in.shape[0] == 1
            and self.tensor_in.shape[1] == 1
        )

    def group_key(self):
        return (self.is_input_ids, self.top_k)


class BatchScheduler:
    def __init__(self, runner, max_batch: int = 8):
        self.runner = runner
        self.max_batch = max(1, max_batch)
        self.queue = queue.SimpleQueue()
        self.batches = 0
        self.batched_jobs = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, tensor_in, is_input_ids, session_id, use_cache, top_k):
        job = Job(tensor_in, is_input_ids, session_id, use_cache, top_k)
        self.queue.put(job)
        job.done.wait()
        if job.error is not None:
            raise job.error
        return job.result

    def stats(self):
        return {
            "batches": self.batches,
            "batched_jobs": self.batched_jobs,
            "avg_batch_size": (self.batched_jobs / self.batches) if self.batches else 0.0,
        }

    def _drain(self, first):
        pending = [first]
        while len(pending) < self.max_batch:
            try:
                pending.append(self.queue.get_nowait())
            except queue.Empty:
                break
        return pending

    def _loop(self):
        while True:
            try:
                pending = self._drain(self.queue.get())
                self._dispatch(pending)
            except Exception:
                log.exception("batch scheduler loop error")

    def _dispatch(self, pending):
        groups = {}
        solo = []
        for job in pending:
            if job.batchable():
                groups.setdefault(job.group_key(), []).append(job)
            else:
                solo.append(job)

        for job in solo:
            self._run_one(job)

        for (_is_ids, top_k), jobs in groups.items():
            if len(jobs) == 1:
                self._run_one(jobs[0])
                continue
            self._run_group(jobs, top_k)

    def _run_one(self, job):
        try:
            job.result = self.runner.run(
                job.tensor_in,
                is_input_ids=job.is_input_ids,
                session_id=job.session_id,
                use_cache=job.use_cache,
                top_k=job.top_k,
            )
        except Exception as exc:
            job.error = exc
        finally:
            job.done.set()

    def _run_group(self, jobs, top_k):
        try:
            results = self.runner.run_batch(jobs, top_k=top_k)
            self.batches += 1
            self.batched_jobs += len(jobs)
            for job, result in zip(jobs, results):
                job.result = result
        except Exception as exc:
            log.exception("batched forward failed")
            for job in jobs:
                job.error = exc
        finally:
            for job in jobs:
                job.done.set()
