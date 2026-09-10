"""Actor workers must close their nested preview pools before returning."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import subprocess
import sys
import unittest
from unittest.mock import patch

from rl_leader import run_sequential_response_ddqn as runner
from rl_leader.response_dqn_collect import _response_process_pool


def _preview_probe(payload):
    result = _response_process_pool(2).submit(abs, -7).result(timeout=20)
    if payload.get("interrupt"):
        raise InterruptedError("checkpoint saved")
    return {"result": result}


def _actor_probe(interrupt):
    with patch.object(runner, "collect_actor", side_effect=_preview_probe):
        try:
            result = runner._collect_actor_worker({"interrupt": interrupt})
        except InterruptedError:
            result = {"interrupted": True}
    assert not multiprocessing.active_children(), "preview children leaked"
    return result


def run_nested_pool_probe():
    with ProcessPoolExecutor(max_workers=2) as pool:
        success = pool.submit(_actor_probe, False)
        interrupted = pool.submit(_actor_probe, True)
        assert success.result(timeout=40) == {"result": 7}
        assert interrupted.result(timeout=40) == {"interrupted": True}


class TestActorPoolLifecycle(unittest.TestCase):
    def test_cleanup_on_success(self):
        with patch.object(runner, "collect_actor", return_value={"ok": True}), patch.object(
            runner, "_shutdown_response_process_pools"
        ) as shutdown:
            self.assertEqual(runner._collect_actor_worker({}), {"ok": True})
        shutdown.assert_called_once_with(wait=True)

    def test_cleanup_on_error(self):
        for error in (InterruptedError("saved"), ValueError("invalid")):
            with self.subTest(error=type(error).__name__):
                with patch.object(runner, "collect_actor", side_effect=error), patch.object(
                    runner, "_shutdown_response_process_pools"
                ) as shutdown:
                    with self.assertRaises(type(error)):
                        runner._collect_actor_worker({})
                shutdown.assert_called_once_with(wait=True)

    def test_nested_process_pools_exit(self):
        code = (
            "from src.tests.test_sequential_response_pool_lifecycle "
            "import run_nested_pool_probe; "
            "run_nested_pool_probe(); print('nested pools closed')"
        )
        result = subprocess.run(
            [sys.executable, "-B", "-c", code], capture_output=True,
            text=True, timeout=60, check=True,
        )
        self.assertIn("nested pools closed", result.stdout)


if __name__ == "__main__":
    unittest.main()
