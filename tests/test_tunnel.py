import threading

from cua.cloud.vast.tunnel import NullTunnel, SshTunnel


class FakePopen:
    instances = []

    def __init__(self, argv, **kw):
        self.argv, self.rc = argv, None
        FakePopen.instances.append(self)

    def poll(self):
        return self.rc

    def terminate(self):
        self.rc = 0

    def wait(self, timeout=None):
        return self.rc

    def kill(self):
        self.rc = -9


def test_argv_binds_loopback_both_ways():
    tunnel = SshTunnel(
        ["ssh", "-p", "2222", "-i", "k", "root@h"], [("R", 8900), ("L", 8001)], popen=FakePopen, sleep=lambda s: None
    )
    argv = tunnel.argv()
    assert argv[0] == "ssh" and "-N" in argv and argv[-1] == "root@h"
    assert (
        argv[argv.index("-R") + 1] == "127.0.0.1:8900:127.0.0.1:8900"
        and argv[argv.index("-L") + 1] == "127.0.0.1:8001:127.0.0.1:8001"
    )
    assert "ServerAliveInterval=15" in argv and "ExitOnForwardFailure=yes" in argv


def test_supervisor_restarts_until_closed():
    FakePopen.instances.clear()
    restarted = threading.Event()

    def sleep(seconds):
        if len(FakePopen.instances) >= 2:
            restarted.set()

    with SshTunnel(["ssh", "root@h"], [("R", 8900)], popen=FakePopen, sleep=sleep) as tunnel:
        FakePopen.instances[0].rc = 255
        assert restarted.wait(2)
    assert not tunnel.alive and all(p.rc is not None for p in FakePopen.instances)


def test_null_tunnel_is_a_noop():
    with NullTunnel([("R", 8900)]) as t:
        assert t.forwards == [("R", 8900)]
