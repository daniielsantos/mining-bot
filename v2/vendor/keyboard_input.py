from __future__ import annotations

import ctypes
import ctypes.wintypes
import sys
import time

INPUT_KEYBOARD = 1
INPUT_MOUSE = 0
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010
MOUSEEVENTF_MOVE_NOCOALESCE = 0x2000
MAPVK_VK_TO_VSC = 0

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong
_extra_info = ULONG_PTR(0)

VK_MAP = {
    "a": 0x41,
    "d": 0x44,
    "e": 0x45,
    "s": 0x53,
    "w": 0x57,
    "space": 0x20,
}

IS_WINDOWS = sys.platform == "win32"
_pressed_vk: set[int] = set()

if IS_WINDOWS:
    _user32 = ctypes.windll.user32
else:
    _user32 = None


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.wintypes.DWORD),
        ("wParamL", ctypes.wintypes.WORD),
        ("wParamH", ctypes.wintypes.WORD),
    ]


class _INPUTUNION(ctypes.Union):
    _fields_ = [
        ("ki", KEYBDINPUT),
        ("mi", MOUSEINPUT),
        ("hi", HARDWAREINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("union", _INPUTUNION),
    ]


def _require_windows() -> None:
    if not IS_WINDOWS:
        raise RuntimeError("SendInput so funciona no Windows.")


def _vk_for(key: str) -> int:
    normalized = key.lower()
    if normalized not in VK_MAP:
        raise ValueError(f"Tecla nao suportada: {key}")
    return VK_MAP[normalized]


def _scan_for_vk(vk: int) -> int:
    _require_windows()
    return int(_user32.MapVirtualKeyW(vk, MAPVK_VK_TO_VSC)) & 0xFF


def _send_input_keyboard(ki: KEYBDINPUT) -> None:
    _require_windows()
    event = INPUT(type=INPUT_KEYBOARD, union=_INPUTUNION(ki=ki))
    sent = _user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
    if sent != 1:
        err = ctypes.windll.kernel32.GetLastError()
        raise OSError(
            f"SendInput falhou (vk={ki.wVk}, scan={ki.wScan}, flags={ki.dwFlags}, err={err})"
        )


def _press_vk(vk: int) -> None:
    _send_input_keyboard(KEYBDINPUT(vk, 0, 0, 0, _extra_info))


def _release_vk(vk: int) -> None:
    _send_input_keyboard(KEYBDINPUT(vk, 0, 0, KEYEVENTF_KEYUP, _extra_info))


def _press_scan(scan: int, vk: int | None = None) -> None:
    try:
        _send_input_keyboard(KEYBDINPUT(0, scan, KEYEVENTF_SCANCODE, 0, _extra_info))
    except OSError:
        if vk is None:
            raise
        _press_vk(vk)


def _release_scan(scan: int, vk: int | None = None) -> None:
    try:
        _send_input_keyboard(
            KEYBDINPUT(0, scan, KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP, 0, _extra_info)
        )
    except OSError:
        if vk is None:
            raise
        _release_vk(vk)


def press_key(key: str) -> None:
    _require_windows()
    vk = _vk_for(key)
    if vk in _pressed_vk:
        return
    _press_scan(_scan_for_vk(vk), vk=vk)
    _pressed_vk.add(vk)


def release_key(key: str) -> None:
    _require_windows()
    vk = _vk_for(key)
    if vk not in _pressed_vk:
        return
    _release_scan(_scan_for_vk(vk), vk=vk)
    _pressed_vk.discard(vk)


def _send_input_mouse(mi: MOUSEINPUT) -> None:
    _require_windows()
    event = INPUT(type=INPUT_MOUSE, union=_INPUTUNION(mi=mi))
    sent = _user32.SendInput(1, ctypes.byref(event), ctypes.sizeof(INPUT))
    if sent != 1:
        err = ctypes.windll.kernel32.GetLastError()
        raise OSError(
            f"SendInput mouse falhou (dx={mi.dx}, dy={mi.dy}, flags={mi.dwFlags}, err={err})"
        )


def mouse_move_relative(dx: int, dy: int = 0, *, nocoalesce: bool = False) -> None:
    """
    Movimento relativo do mouse — jogos em primeira pessoa (GTA) leem isto como olhar.
    dx > 0 = girar câmera para a direita (minimapa/seta acompanham).
    """
    if dx == 0 and dy == 0:
        return
    flags = MOUSEEVENTF_MOVE | (MOUSEEVENTF_MOVE_NOCOALESCE if nocoalesce else 0)
    _send_input_mouse(
        MOUSEINPUT(
            dx=int(dx),
            dy=int(dy),
            mouseData=0,
            dwFlags=flags,
            time=0,
            dwExtraInfo=_extra_info,
        )
    )


def mouse_move_legacy(dx: int, dy: int = 0) -> None:
    """API mouse_event — fallback quando SendInput e ignorado (FiveM/raw input)."""
    if dx == 0 and dy == 0:
        return
    _require_windows()
    _user32.mouse_event(MOUSEEVENTF_MOVE, int(dx), int(dy), 0, 0)


def center_cursor_on_foreground() -> bool:
    """Centraliza cursor na janela em foco — alguns jogos so leem look com cursor no centro."""
    _require_windows()
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return False
    rect = ctypes.wintypes.RECT()
    if not _user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return False
    pt = ctypes.wintypes.POINT(rect.left + rect.right // 2, rect.top + rect.bottom // 2)
    if not _user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return False
    return bool(_user32.SetCursorPos(pt.x, pt.y))


def mouse_move_relative_chunked(
    dx: int,
    dy: int = 0,
    *,
    chunk: int = 18,
    delay_s: float = 0.006,
    backend: str = "sendinput",
) -> None:
    """Divide movimento grande — alguns jogos ignoram deltas altos num unico evento."""
    if dx == 0 and dy == 0:
        return
    remaining_x = int(dx)
    remaining_y = int(dy)
    step = max(4, int(chunk))
    move_fn = mouse_move_legacy if backend == "mouse_event" else mouse_move_relative
    while remaining_x != 0 or remaining_y != 0:
        step_x = max(-step, min(step, remaining_x))
        step_y = max(-step, min(step, remaining_y))
        move_fn(step_x, step_y)
        remaining_x -= step_x
        remaining_y -= step_y
        if delay_s > 0 and (remaining_x != 0 or remaining_y != 0):
            time.sleep(delay_s)


def mouse_camera_look(
    dx: int,
    dy: int = 0,
    *,
    hold_rmb: bool = False,
    backend: str = "auto",
    center_cursor: bool = True,
) -> str:
    """
    Look horizontal. Retorna backend usado.
    backend: auto | sendinput | mouse_event | both
    """
    if dx == 0 and dy == 0:
        return "none"
    if center_cursor:
        center_cursor_on_foreground()
        time.sleep(0.015)
    if hold_rmb:
        _send_input_mouse(MOUSEINPUT(0, 0, 0, MOUSEEVENTF_RIGHTDOWN, 0, _extra_info))
        time.sleep(0.025)

    used = "sendinput"
    if backend == "sendinput":
        mouse_move_relative_chunked(dx, dy, backend="sendinput")
        used = "sendinput"
    elif backend == "mouse_event":
        mouse_move_relative_chunked(dx, dy, backend="mouse_event")
        used = "mouse_event"
    elif backend == "both":
        mouse_move_relative_chunked(dx, dy, backend="sendinput")
        mouse_move_relative_chunked(dx, dy, backend="mouse_event")
        used = "both"
    else:
        mouse_move_relative_chunked(dx, dy, backend="sendinput")
        half = int(round(dx * 0.5))
        if half != 0:
            mouse_move_relative_chunked(half, 0, backend="mouse_event")
        used = "auto"

    if hold_rmb:
        time.sleep(0.025)
        _send_input_mouse(MOUSEINPUT(0, 0, 0, MOUSEEVENTF_RIGHTUP, 0, _extra_info))
    return used


def release_all_keys() -> None:
    _require_windows()
    for vk in list(_pressed_vk):
        _release_scan(_scan_for_vk(vk), vk=vk)
    _pressed_vk.clear()


def tap_key(key: str, hold_ms: float = 50.0) -> None:
    """Pulso curto — nunca deixa tecla marcada em _pressed_vk."""
    _require_windows()
    vk = _vk_for(key)
    scan = _scan_for_vk(vk)
    if vk in _pressed_vk:
        _release_scan(scan, vk=vk)
        _pressed_vk.discard(vk)
    _press_scan(scan, vk=vk)
    time.sleep(max(hold_ms, 0) / 1000.0)
    _release_scan(scan, vk=vk)
    _pressed_vk.discard(vk)


def is_key_pressed(key: str) -> bool:
    return _vk_for(key) in _pressed_vk


def is_physical_key_down(key: str) -> bool:
    """Estado real do teclado (funciona com foco no jogo)."""
    if not IS_WINDOWS:
        return False
    vk = _vk_for(key)
    return bool(_user32.GetAsyncKeyState(vk) & 0x8000)


def get_foreground_window_title() -> str:
    if not IS_WINDOWS:
        return ""
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return ""
    length = _user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    _user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value


# Titulos como "RAGЕ Multiрlауеr" usam homoglifos cirilicos (Е, р, а, у).
_HOMOGLYPH_MAP = str.maketrans(
    {
        "\u0410": "a",
        "\u0430": "a",
        "\u0412": "b",
        "\u0432": "b",
        "\u0415": "e",
        "\u0435": "e",
        "\u0401": "e",
        "\u0451": "e",
        "\u041a": "k",
        "\u043a": "k",
        "\u041c": "m",
        "\u043c": "m",
        "\u041d": "h",
        "\u043d": "h",
        "\u041e": "o",
        "\u043e": "o",
        "\u0420": "p",
        "\u0440": "p",
        "\u0421": "c",
        "\u0441": "c",
        "\u0422": "t",
        "\u0442": "t",
        "\u0423": "y",
        "\u0443": "y",
        "\u0425": "x",
        "\u0445": "x",
        "\u0456": "i",
        "\u0454": "e",
    }
)


def normalize_window_title(title: str) -> str:
    return title.translate(_HOMOGLYPH_MAP).lower()


def get_foreground_process_name() -> str:
    if not IS_WINDOWS:
        return ""
    hwnd = _user32.GetForegroundWindow()
    if not hwnd:
        return ""
    pid = ctypes.wintypes.DWORD()
    _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    kernel32 = ctypes.windll.kernel32
    access = 0x1000  # PROCESS_QUERY_LIMITED_INFORMATION
    handle = kernel32.OpenProcess(access, False, pid.value)
    if not handle:
        return ""
    try:
        buf = ctypes.create_unicode_buffer(260)
        size = ctypes.wintypes.DWORD(260)
        if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return buf.value.rsplit("\\", 1)[-1].lower()
    finally:
        kernel32.CloseHandle(handle)
    return ""


def is_game_foreground(
    keywords: list[str] | None = None,
    process_names: list[str] | None = None,
) -> bool:
    title = normalize_window_title(get_foreground_window_title())
    if title:
        defaults = (
            "grand theft auto",
            "fivem",
            "citizenfx",
            "gta",
            "rage",
            "multiplayer",
            "grand rp",
        )
        for keyword in keywords or defaults:
            if normalize_window_title(keyword) in title:
                return True

    process = get_foreground_process_name()
    if process:
        defaults_proc = ("gta5.exe", "gtav.exe", "playgtav.exe", "fivem.exe", "citizenfx.exe")
        for name in process_names or defaults_proc:
            if process == name.lower():
                return True
    return False
