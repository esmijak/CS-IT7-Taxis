from schemas import registerTable, Table
from collections import deque

CACHED_TIMESLOT_COUNT = 3
DEMAND_CACHE_DEBUG = False

# TODO: Make fetching queued pages asynchronous (promises?)

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

"""

class DemandCache:
    class _Impl:
        def __init__(self, spark, sqlCtx):
            registerTable(sqlCtx, Table.DEMAND)
            self._spark = spark
            self._start_tid = -1
            self._current_df = None
            self._next_dfs = deque([])
            if DEMAND_CACHE_DEBUG:
                 print('Demand cache initialized')

        def _fetch(self, tid):
            return self._spark.sql('SELECT * FROM demand WHERE pickup_timeslot_id = {}'.format(tid))

        def _reinit(self, tid):
            if DEMAND_CACHE_DEBUG:
                print('Reinitializing cache, fetching tids {} through {}'.format(tid, tid + CACHED_TIMESLOT_COUNT - 1))
            self._start_tid = tid
            self._current_df = self._fetch(self._start_tid)
            for i in range(1, CACHED_TIMESLOT_COUNT):
                self._next_dfs.append(self._fetch(tid + i))
            if DEMAND_CACHE_DEBUG:
                print('Reinitialization done. tid {} in memory, next {} in queue'.format(tid, len(self._next_dfs)))

        def _update_df(self, tid, cid):
            if self._current_df is None:
                self._reinit(tid)
            elif tid == self._start_tid:
                if DEMAND_CACHE_DEBUG:
                    print('DemandCache: Cache hit, tid {}'.format(tid))
            elif tid == self._start_tid + 1:
                if len(self._next_dfs) > 0:
                    if DEMAND_CACHE_DEBUG:
                        print('Updating cache: Discarding tid {}, adding {} to the queue'.format(self._start_tid,
                                                                                                 tid + CACHED_TIMESLOT_COUNT - 1))
                    next = self._next_dfs.popleft()
                    del self._current_df
                    self._current_df = next
                    self._start_tid = tid
                    self._next_dfs.append(self._fetch(tid + CACHED_TIMESLOT_COUNT - 1))
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
            self._update_df(tid, cid)
            res = self._current_df.filter('pickup_timeslot_id = {} AND pickup_cid = {}'.format(tid, cid))
            return res.head()['cnt'] if res.count() > 0 else 0

        def hint(self, tid):
            self._update_df(tid, 0)

        def clear(self):
            del self._current_df
            self._current_df = None
            for df in self._next_dfs:
                del df
            self._next_dfs = deque([])
            self._start_tid = -1

    instance = None
    def init(self, spark, sqlCtx):
        if DemandCache.instance is None:
            DemandCache.instance = DemandCache._Impl(spark, sqlCtx)
    def __getattr__(self, item):
        return getattr(DemandCache.instance, item)


demandCache = DemandCache()