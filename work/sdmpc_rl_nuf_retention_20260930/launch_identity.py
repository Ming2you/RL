"""Small stdlib-only Windows identity checks; no enumeration or numerical imports."""
import os
import sys


def probe_process(pid):
    """Query one PID without signals; creation time distinguishes reused PIDs."""
    if os.name != "nt" or type(pid) is not int or not 0 < pid <= 0xffffffff:
        return "unknown", None
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)]*4
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return ("dead" if ctypes.get_last_error() == 87 else "unknown"), None
    try:
        created, exited, cpu, user = (wintypes.FILETIME() for _ in range(4))
        code = wintypes.DWORD()
        if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited),
                                      ctypes.byref(cpu), ctypes.byref(user)):
            return "unknown", None
        if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
            return "unknown", None
        return ("live" if code.value == 259 else "dead"), (created.dwHighDateTime << 32) | created.dwLowDateTime
    finally:
        kernel.CloseHandle(handle)


def verify_claim(launcher, worker, owner, parent_pid, probe=probe_process):
    for record in (launcher, worker, owner):
        if (record is None or type(record.get("created")) is not int or
                probe(record["pid"]) != ("live", record["created"])):
            raise ValueError("Launch reservation process identity is dead, reused or UNKNOWN")
    if not owner["created"] <= launcher["created"] <= worker["created"]:
        raise ValueError("Launch reservation creation order differs")
    if worker["pid"] == launcher["pid"]:
        if parent_pid != owner["pid"]:
            raise ValueError("Direct worker parent differs from launch owner")
        return "direct"
    # Only the Windows venv redirector's immediate child is admissible.
    if (os.name != "nt" or sys.executable == getattr(sys, "_base_executable", sys.executable) or
            parent_pid != launcher["pid"]):
        raise ValueError("Redirector worker parent differs from bound launcher")
    return "windows-venv-redirector"
