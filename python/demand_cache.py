from schemas import registerTable, Table
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from timeit import timeit

CACHED_TIMESLOT_COUNT = 8              # The total number of pages to be fetched, including the 'current page'

DEMAND_CACHE_DEBUG = True              # Controls debug prints

DEMAND_CACHE_ASYNC = True              # If true, future pages will be fetched asynchronously

DEMAND_CACHE_UNPACK_DATAFRAMES = True  # If true, dataframes will be converted to a dictionary format.
                                       # This will make queries faster, but initial fetching slower.
                                       # If both DEMAND_CACHE_ASYNC and DEMAND_CACHE_HEAVY_FETCH
                                       # are true, this conversion will be done in the background if possible

"""
DemandCache can be used to efficiently fetch individual values from the demand table.
(Table.DEMAND, or hdfs://csit7-master/user/csit7/demand)
The cache assumes that you are going to be querying the same timeslot many times in a row, like this:

  for tid in range(0, 10000):
    for cid in range(0, 10000):
      demand = demandCache.get_demand(tid, cid)
      # do something with demand

If this is not the case, you should probably not use this cache, but rather query the table directly.

Before using the cache in any way, you MUST call 'demandCache.init(spark, sqlCtx)'
After that, you can do these things with the global variable demandCache:
    get_demand(timeslot_id, cluster_id)
        Returns the demand for that timeslot and cluster. Also updates the cache if required.
    hint(timeslot_id)
        Indicates that you are planning to query the given timeslot, and will prepare the cache accordingly.
        This is not required, since get_demand will update it anyway, but calling it will make the runtime
        of get_demand more predictable.
    clear()
        Clears all data from the cache. The next time you call get_demand or hint will take longer than usual
        since it will require rebuilding the cache from scratch. Use this for cleanup in order to prevent
        memory issues.

The constants above this comment can be adjusted, but you probably shouldn't.
CACHED_TIMESLOT_COUNT can be set higher if you are processing a lot of data, in the proper order.
DEMAND_CACHE_DEBUG = True will just flood your console.
DEMAND_CACHE_ASYNC = False will make cache misses much more expensive, while only 
                           providing a small increase to performance for hits.
DEMAND_CACHE_UNPACK_DATAFRAMES = False will destroy performance. It only makes sense if you are not processing
                                       data sequentially, but then you shouldn't be using the cache anyway.

"""

class DemandCache:
    class _Impl:
        def __init__(self, spark, sqlCtx):
            registerTable(sqlCtx, Table.DEMAND)
            self._spark = spark
            self._exec = ThreadPoolExecutor(max_workers=3)
            self._start_tid = -1
            self._current_page = None
            self._next_pages = deque([])
            if DEMAND_CACHE_DEBUG:
                 print('Demand cache initialized')

        def _fetch(self, tid):
            res = self._spark.sql('SELECT * FROM demand WHERE pickup_timeslot_id = {}'.format(tid))
            if DEMAND_CACHE_DEBUG:
                print("Page {} fetched".format(tid))
            if DEMAND_CACHE_UNPACK_DATAFRAMES:
                res = {cid: demand for (_, cid, demand) in res.collect()}
            return res

        def _reinit(self, tid):
            if DEMAND_CACHE_DEBUG:
                print('Reinitializing cache, fetching tids {} through {}'.format(tid, tid + CACHED_TIMESLOT_COUNT - 1))
            self._start_tid = tid
            self._current_page = self._fetch(self._start_tid)
            for page in self._next_pages:
                del page
            self._next_pages = deque([])
            for i in range(1, CACHED_TIMESLOT_COUNT):
                if DEMAND_CACHE_ASYNC:
                    self._next_pages.append(self._exec.submit(self._fetch, tid + i))
                else:
                    self._next_pages.append(self._fetch(tid + i))
            if DEMAND_CACHE_DEBUG:
                print('Reinitialization done. tid {} in memory, next {} in queue'.format(tid, len(self._next_pages)))

        def _update_page(self, tid, cid):
            if self._current_page is None:
                self._reinit(tid)
            elif tid == self._start_tid:
                if DEMAND_CACHE_DEBUG:
                    print('DemandCache: Cache hit, tid {}'.format(tid))
            elif tid == self._start_tid + 1:
                if len(self._next_pages) > 0:
                    if DEMAND_CACHE_DEBUG:
                        print('Updating cache: Discarding tid {}, adding {} to the queue'.format(self._start_tid,
                                                                                                 tid + CACHED_TIMESLOT_COUNT - 1))
                    next = self._next_pages.popleft()
                    del self._current_page
                    self._current_page = next.result() if DEMAND_CACHE_ASYNC else next
                    self._start_tid = tid
                    if DEMAND_CACHE_ASYNC:
                        self._next_pages.append(self._exec.submit(self._fetch, tid + CACHED_TIMESLOT_COUNT - 1))
                    else:
                        self._next_pages.append(self._fetch(tid + CACHED_TIMESLOT_COUNT - 1))
                else:
                    print('DemandCache: Queue exhausted')
                    self._reinit(tid)
            else:
                if DEMAND_CACHE_DEBUG:
                    print('DemandCache: Cache miss, had {} through {}, {} requested'.format(self._start_tid,
                                                                                            self._start_tid + CACHED_TIMESLOT_COUNT - 1,
                                                                                            tid))
                self._reinit(tid)

        def get_demand(self, tid, cid):
            self._update_page(tid, cid)
            if DEMAND_CACHE_UNPACK_DATAFRAMES:
                return self._current_page.get(cid, 0)
            else:
                res = self._current_page.filter('pickup_timeslot_id = {} AND pickup_cid = {}'.format(tid, cid))
                return res.head()['cnt'] if res.count() > 0 else 0

        def hint(self, tid):
            self._update_page(tid, 0)

        def clear(self):
            del self._current_page
            self._current_page = None
            for page in self._next_pages:
                del page
            self._next_pages = deque([])
            self._start_tid = -1

    instance = None
    def init(self, spark, sqlCtx):
        if DemandCache.instance is None:
            DemandCache.instance = DemandCache._Impl(spark, sqlCtx)
    def __getattr__(self, item):
        return getattr(DemandCache.instance, item)


demandCache = DemandCache()
def demandCacheStressTest(page_count, cluster_count):
    global demandCache
    def fn():
        for tid in range(1, page_count):
            for cid in range(1, cluster_count):
                demandCache.get_demand(tid, cid)
    print(timeit(fn, number=3))
