import concurrent.futures as futures
import multiprocessing
import threading
import time
import unittest


def process_probe(job):
    import os
    return {'job_id': job['job_id'], 'pid': os.getpid()}


class ParallelExtractionTests(unittest.TestCase):
    def test_quota_limits_unique_jobs_and_long_first(self):
        from training.dmd.parallel import dispatch
        jobs = [dict(job_id=str(i), subset=s, video_size_bytes=i+1)
                for i, s in enumerate(['distraction']*7+['drowsiness']*4+['gaze']*3)]
        active = dict.fromkeys(('distraction', 'drowsiness', 'gaze'), 0)
        peak = active.copy(); order = []; lock = threading.Lock()
        def work(job):
            with lock:
                s = job['subset']; active[s] += 1; peak[s] = max(peak[s], active[s])
            time.sleep(.02)
            with lock: active[s] -= 1
            return {'job_id': job['job_id']}
        with futures.ThreadPoolExecutor(4) as executor:
            def submit(job):
                order.append(job)
                return executor.submit(work, job)
            results, pending = dispatch(jobs, dict(distraction=2,drowsiness=1,gaze=1), submit)
        self.assertEqual(len({r['job_id'] for r in results}), len(jobs))
        self.assertEqual(pending, [])
        self.assertEqual(peak, dict(distraction=2,drowsiness=1,gaze=1))
        self.assertEqual([j['job_id'] for j in order if j['subset']=='distraction'][0], '6')

    def test_stop_drains_active_without_submitting_rest(self):
        from training.dmd.parallel import dispatch
        stop = threading.Event()
        jobs = [dict(job_id=str(i),subset='gaze',video_size_bytes=i) for i in range(5)]
        def work(job):
            stop.set()
            return {'job_id':job['job_id']}
        with futures.ThreadPoolExecutor(1) as executor:
            results, pending = dispatch(jobs, dict(distraction=1,drowsiness=1,gaze=1),
                                        lambda j:executor.submit(work,j), stop.is_set)
        self.assertEqual(len(results), 1)
        self.assertEqual(len(pending), 4)

    def test_exception_is_reported_and_remaining_jobs_continue(self):
        from training.dmd.parallel import dispatch
        jobs = [dict(job_id=str(i),subset='gaze',video_size_bytes=i) for i in range(3)]
        def work(job):
            if job['job_id']=='1': raise ValueError('intentional failure')
            return {'job_id':job['job_id'],'status':'completed'}
        with futures.ThreadPoolExecutor(1) as executor:
            results, pending = dispatch(jobs, dict(distraction=1,drowsiness=1,gaze=1),
                                        lambda j:executor.submit(work,j))
        self.assertEqual(len(results), 3)
        self.assertEqual(sum(r['status']=='failed' for r in results), 1)
        self.assertEqual(pending, [])

    def test_spawn_process_contract(self):
        from training.dmd.parallel import dispatch
        import os
        jobs = [dict(job_id=str(i),subset='gaze',video_size_bytes=i) for i in range(2)]
        with futures.ProcessPoolExecutor(2,mp_context=multiprocessing.get_context('spawn')) as executor:
            results, pending = dispatch(jobs, dict(distraction=1,drowsiness=1,gaze=2),
                                        lambda j:executor.submit(process_probe,j))
        self.assertFalse(pending)
        self.assertTrue(all(r['pid']!=os.getpid() for r in results))

    def test_invalid_quotas_and_duplicate_jobs_rejected(self):
        from training.dmd.parallel import dispatch, validate_quotas
        for q in ({'gaze':2},dict(distraction=0,drowsiness=1,gaze=1),dict(distraction=True,drowsiness=1,gaze=1)):
            with self.assertRaises(ValueError): validate_quotas(q)
        j=dict(job_id='x',subset='gaze',video_size_bytes=1)
        with self.assertRaises(ValueError): dispatch([j,j],dict(distraction=1,drowsiness=1,gaze=1),None)
