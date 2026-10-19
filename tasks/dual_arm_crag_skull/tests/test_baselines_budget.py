import json
import multiprocessing
import os

import pytest

from tasks.dual_arm_crag_skull.baselines.budget import BudgetExhausted, BudgetPool


def test_a_pool_is_created_once_with_its_total(tmp_path):
    pool = BudgetPool.create(tmp_path / 'pool', 100.0, note='practice round')
    assert pool.status()['total_usd'] == 100.0
    with pytest.raises(FileExistsError):
        BudgetPool.create(tmp_path / 'pool', 200.0, note='a second pool in the same place')
    with pytest.raises(FileNotFoundError):
        BudgetPool(tmp_path / 'elsewhere')
    for bad in (0, -1, float('inf'), float('nan'), True):
        with pytest.raises(ValueError):
            BudgetPool.create(tmp_path / f'bad{bad}', bad, note='x')


def test_a_call_is_made_only_if_its_worst_case_fits_in_what_is_left(tmp_path):
    pool = BudgetPool.create(tmp_path / 'pool', 10.0, note='t')
    first = pool.reserve(6.0, who='dexagent:t1', what='turn 1')
    with pytest.raises(BudgetExhausted):
        pool.reserve(4.5, who='aspire:t1', what='turn 1')  # 6 reserved + 4.5 > 10, though nothing is spent yet
    pool.settle(first, 1.25, outcome='call')
    second = pool.reserve(8.0, who='aspire:t1', what='turn 1')
    status = pool.status()
    assert status['spent_usd'] == pytest.approx(1.25)
    assert status['reserved_usd'] == pytest.approx(8.0)
    assert status['left_usd'] == pytest.approx(0.75)
    pool.settle(second, 0.0, outcome='not_sent')
    assert pool.status()['left_usd'] == pytest.approx(8.75)


def test_spending_is_counted_per_baseline_and_logged(tmp_path):
    pool = BudgetPool.create(tmp_path / 'pool', 100.0, note='t')
    for who, cost in (('dexagent:a', 2.0), ('aspire:b', 0.5), ('dexagent:a', 1.0)):
        pool.settle(pool.reserve(5.0, who=who, what='call'), cost, outcome='call')
    status = pool.status()
    assert status['by_who'] == {'dexagent:a': pytest.approx(3.0), 'aspire:b': pytest.approx(0.5)}
    assert status['calls'] == 3
    events = [json.loads(line)['event'] for line in (tmp_path / 'pool' / 'pool.jsonl').read_text().splitlines()]
    assert events == ['created'] + ['reserve', 'settle'] * 3


def test_a_call_dearer_than_its_reservation_is_charged_in_full_and_flagged_and_settled_once(tmp_path):
    pool = BudgetPool.create(tmp_path / 'pool', 2.0, note='t')
    reservation = pool.reserve(1.0, who='w', what='x')
    assert pool.settle(reservation, 0.4, outcome='call') == {'exceeded': False}
    with pytest.raises(KeyError):
        pool.settle(reservation, 0.4, outcome='call')
    reservation = pool.reserve(1.0, who='w', what='y')
    # the money is spent whatever the bound said: recorded as it was, and the caller is told to stop
    assert pool.settle(reservation, 1.5, outcome='call') == {'exceeded': True}
    status = pool.status()
    assert status['spent_usd'] == pytest.approx(1.9) and status['left_usd'] == pytest.approx(0.1)
    events = [json.loads(line) for line in (tmp_path / 'pool' / 'pool.jsonl').read_text().splitlines()]
    assert events[-1]['exceeded_reservation'] is True


def _reserve_and_die(directory):
    BudgetPool(directory).reserve(3.0, who='crashed', what='call in flight')
    os._exit(0)


def test_a_reservation_of_a_process_that_died_is_charged_at_its_worst_case(tmp_path):
    pool = BudgetPool.create(tmp_path / 'pool', 10.0, note='t')
    process = multiprocessing.get_context('spawn').Process(target=_reserve_and_die, args=(tmp_path / 'pool',))
    process.start()
    process.join()
    status = pool.status()  # the dead process's call may have been billed
    assert status['spent_usd'] == pytest.approx(3.0) and status['reserved_usd'] == 0.0
    assert status['by_who'] == {'crashed': pytest.approx(3.0)}


def _spend(directory, n, queue):
    try:
        pool = BudgetPool(directory)
        made = 0
        for _ in range(n):
            try:
                reservation = pool.reserve(1.0, who=f'p{os.getpid()}', what='call')
            except BudgetExhausted:
                continue
            pool.settle(reservation, 1.0, outcome='call')
            made += 1
        queue.put(made)
    except BaseException as exc:  # reported to the test, which fails instead of waiting for a result forever
        queue.put(f'{type(exc).__name__}: {exc}')


def test_concurrent_runs_cannot_overspend_together(tmp_path):
    BudgetPool.create(tmp_path / 'pool', 25.0, note='t')
    context = multiprocessing.get_context('spawn')
    queue = context.Queue()
    workers = [context.Process(target=_spend, args=(tmp_path / 'pool', 20, queue)) for _ in range(4)]
    for worker in workers:
        worker.start()
    results = [queue.get(timeout=120) for _ in workers]
    for worker in workers:
        worker.join(timeout=10)
    assert all(isinstance(made, int) for made in results), results
    status = BudgetPool(tmp_path / 'pool').status()
    assert sum(results) == 25 and status['spent_usd'] == pytest.approx(25.0) and status['left_usd'] == pytest.approx(0.0)


def test_the_cli_makes_the_configured_pool_once_and_reports_it(tmp_path, capsys):
    from tasks.dual_arm_crag_skull.baselines.__main__ import main
    assert main(['budget', 'create', '--pool', str(tmp_path / 'pool'), '--note', 'practice round']) == 0
    assert BudgetPool(tmp_path / 'pool').status()['total_usd'] == 100.0  # the total comes from configs/jing.json
    with pytest.raises(FileExistsError):
        main(['budget', 'create', '--pool', str(tmp_path / 'pool'), '--note', 'again'])
    capsys.readouterr()
    assert main(['budget', 'status', '--pool', str(tmp_path / 'pool')]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status['left_usd'] == 100.0 and status['spent_usd'] == 0.0
