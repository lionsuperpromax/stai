#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
local_agent.py - локальный движок LLM с ИИ-агентом в стиле Claude Code.

ЭТО СБОРКА «ВСЁ В ОДНОМ ФАЙЛЕ»: не требует scripts/stai.py рядом.
Скопируйте этот файл куда угодно и запустите: python local_agent.py

Что делает:
  1. Проверяет, установлен ли llama.cpp (llama-server), и при необходимости
     скачивает готовую сборку под вашу ОС.
  2. Спрашивает ссылку на модель Hugging Face (прямая ссылка на .gguf или
     страница репозитория - тогда покажет список файлов на выбор),
     скачивает её с прогресс-баром и возможностью продолжить прерванную загрузку.
  3. Запускает llama-server и открывает ТЕРМИНАЛЬНЫЙ чат с агентом, который умеет:
       read_file / write_file / edit_file / apply_patch / list_dir / glob /
       grep / run_shell / web_fetch / todo_write
     Все изменения файлов записываются в журнал и откатываются командой /undo.

Зависимости: только стандартная библиотека Python 3.9+ (Windows / Linux / macOS).
Необязательные пакеты (psutil, certifi) ставятся автоматически при первом запуске.

Запуск ОДНОЙ командой:
    python local_agent.py        # первый запуск: выбор папки, установка, команда stai
    stai                         # все следующие запуски - из любого каталога

При первом запуске программа:
  - спрашивает ОДНУ папку программы и запоминает её;
  - создаёт в ней структуру: llama.cpp, models, undo, logs, workspace, venv;
  - ставит необязательные зависимости в venv и перезапускается уже в нём;
  - прописывает команду stai в PATH (Windows: %LOCALAPPDATA%\\Microsoft\\WindowsApps;
    если запись недоступна - своя папка плюс PATH пользователя; резерв - профиль PowerShell).

Запуск вручную:
    python local_agent.py --set-home       # заново выбрать папку программы
    python local_agent.py --home D:/ai     # указать папку разово и запомнить
    python local_agent.py --server-only    # поднять llama-server как API, без чата
    python local_agent.py --no-bootstrap   # ничего не ставить и не прописывать

Команды в чате: /help  /tools  /undo  /history  /files  /cwd <путь>
                /yes  /ask  /new  /clear  /stats  /server  /mascot  /exit

Где всё лежит (единая папка программы; выбор запоминается в
.local-agent-home.json рядом со скриптом):
    Windows: %APPDATA%\\local-agent  (по умолчанию; либо папка рядом со скриптом)
    Linux/macOS: ~/.local/share/local-agent
    Переопределить: --home, $LOCAL_AGENT_HOME, --set-home
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import platform
import random
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
#                              КОНСТАНТЫ / ОБЩЕЕ
# --------------------------------------------------------------------------- #

APP_NAME = "local-agent"
VERSION = "0.1"
USER_AGENT = f"{APP_NAME}/{VERSION} (+python urllib)"
IS_WINDOWS = os.name == "nt"
IS_LINUX = sys.platform.startswith("linux")
IS_MAC = sys.platform == "darwin"

HF = "https://huggingface.co"
LLAMA_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
LLAMA_RELEASES_PAGE = "https://github.com/ggml-org/llama.cpp/releases/latest/download"
LLAMA_RELEASES_ALL = "https://github.com/ggml-org/llama.cpp/releases"
LLAMA_BUILD_PAGE = "https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md"

MAX_READ_CHARS = 120_000      # сколько символов файла отдавать модели за раз
MAX_TOOL_RESULT = 200_000     # общий предел текста в одном ответе инструмента
MAX_AGENT_STEPS = 60          # максимум итераций агента на один ход
SHELL_TIMEOUT = 180           # секунд на команду run_shell
MAX_HISTORY_MESSAGES = 60     # сколько сообщений держим в контексте

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")

SYSTEM_PROMPT = """\
Ты — автономный ассистент-инженер, работающий прямо в терминале пользователя.
У тебя есть доступ к инструментам, чтобы читать, создавать и изменять файлы,
искать по коду, выполнять команды в оболочке и скачивать страницы из интернета.

Правила работы:
- Чтобы что-то сделать, ты ОБЯЗАН вернуть ровно один блок с вызовом инструмента
  в таком формате:

  <tool_call>
  {"name": "read_file", "arguments": {"path": "src/app.py"}}
  </tool_call>

  Ничего, кроме этого блока (допустимы короткая мысль перед ним и несколько
  блоков подряд, если нужны независимые вызовы). Без инструментов ты не узнаешь
  содержимое файлов — догадки запрещены.
- Несколько вызовов, которые не зависят друг от друга, можно отправить в одном
  ответе: каждый в своём блоке <tool_call>.
- Доступные инструменты: {tools}

Правила безопасной работы:
- Сначала исследуй проект (list_dir, glob, grep, read_file), потом действуй.
- Для правки существующих файлов предпочитай edit_file или apply_patch,
  write_file — для новых файлов и полной перезаписи.
- Перед завершением проверяй результат: запускай тесты/сборку через run_shell,
  если это возможно.
- Не удаляй чужие данные без явной просьбы, не запускай необратимых команд.
- Для задач из нескольких шагов используй todo_write, чтобы отмечать прогресс.
- Файлы относительны к рабочей папке проекта.
- Когда задача выполнена, кратко отчитайся: что сделано, какие файлы затронуты,
  что осталось. Не вызывай инструменты в финальном ответе.
- Отвечай на языке пользователя, компактно и по делу.
"""

# --------------------------------------------------------------------------- #
#                                   ЦВЕТА
# --------------------------------------------------------------------------- #

def enable_vt_mode() -> bool:
    """Включает escape-последовательности (цвета) в консоли Windows 10+."""
    if IS_WINDOWS and _live_terminal():
        try:
            import ctypes
            k32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            for handle in (-11, -12):
                h = k32.GetStdHandle(handle)
                mode = ctypes.c_uint32()
                if k32.GetConsoleMode(h, ctypes.byref(mode)):
                    k32.SetConsoleMode(h, mode.value | 0x0004)
        except Exception:
            return False
    return True


def _live_terminal() -> bool:
    try:
        return sys.stdout is not None and sys.stdout.isatty()
    except Exception:
        return False


class C:
    """Цвета. При отсутствии TTY или NO_COLOR выводятся пустые строки."""

    enabled = bool(_live_terminal()) and not os.environ.get("NO_COLOR")

    # заготовки на случай вызова _colors() до инициализации атрибутов
    RESET = BOLD = DIM = ITALIC = UNDERLINE = ""
    RED = GREEN = YELLOW = BLUE = MAGENTA = CYAN = WHITE = GREY = ""
    BRED = BGREEN = BYELLOW = BBLUE = BMAGENTA = BCYAN = BWHITE = ""

    @classmethod
    def _w(cls, code: str) -> str:
        return f"\x1b[{code}m" if cls.enabled else ""

    def __new__(cls, *_a: Any, **_k: Any) -> "C":  # статический класс
        raise TypeError("C не инстанцируется")


def _colors() -> None:
    names = {
        "RESET": "0", "BOLD": "1", "DIM": "2", "ITALIC": "3", "UNDERLINE": "4",
        "RED": "31", "GREEN": "32", "YELLOW": "33", "BLUE": "34",
        "MAGENTA": "35", "CYAN": "36", "WHITE": "37", "GREY": "90",
        "BRED": "91", "BGREEN": "92", "BYELLOW": "93", "BBLUE": "94",
        "BMAGENTA": "95", "BCYAN": "96", "BWHITE": "97",
    }
    for k, v in names.items():
        setattr(C, k, C._w(v))


_colors()


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


def _win_size() -> Tuple[int, int]:
    try:
        size = shutil.get_terminal_size((100, 30))
        return max(40, size.columns), size.lines
    except Exception:
        return 100, 30


def _disp_width(s: str) -> int:
    w = 0
    for ch in s:
        if unicodedata.east_asian_width(ch) in ("W", "F"):
            w += 2
        else:
            w += 1
    return w


def _clip(s: str, width: int) -> str:
    out = []
    w = 0
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > width:
            break
        out.append(ch)
        w += cw
    return "".join(out)


def _tint(text: str, color: str) -> str:
    return f"{color}{text}{C.RESET}"


def banner() -> None:
    art = r"""
  _                _     _                _   
 | |   ___   ___  | |   / \   __ _  ___ _ __ | |_ 
 | |  / _ \ / __| | |  / _ \ / _` |/ _ \ '_ \| __|
 | |_| (_) | (__  | | / ___ \ (_| |  __/ | | | |_ 
 |_(_)\___/ \___| |_|/_/   \_\__, |\___|_| |_|\__|
                             |___/                """
    w, _ = _win_size()
    print(_tint(art, C.BCYAN))
    title = f"{APP_NAME} v{VERSION} — локальные модели + ИИ-агент"
    print(_tint(title, C.BOLD))
    print(_tint("─" * min(w, _disp_width(title)), C.GREY))
    print()


def rule(title: str = "") -> None:
    w, _ = _win_size()
    if title:
        print(f"\n{C.BCYAN}── {title} " + "─" * max(0, w - _disp_width(title) - 5) + C.RESET)
    else:
        print(f"{C.GREY}" + "─" * min(w, 100) + C.RESET)


def note(msg: str) -> None:
    print(f"{C.BBLUE}ℹ{C.RESET} {msg}")


def ok(msg: str) -> None:
    print(f"{C.BGREEN}✔{C.RESET} {msg}")


def warn(msg: str) -> None:
    print(f"{C.BYELLOW}⚠{C.RESET} {msg}")


def err(msg: str) -> None:
    print(f"{C.BRED}✖{C.RESET} {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
#                                 ASCII-ГРАФИКА
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
#                          МАСКОТ «НЕЙРО» 🐱
# --------------------------------------------------------------------------- #

MASCOT_NAME = "Нейро"


class Mascot:
    """Рисует и анимирует маскота Нейро в терминале."""

    FACES = {
        "idle":    "=^.^=",
        "happy":   "=^o^=",
        "wink":    "=^-^=",
        "think":   "=.-.=",
        "works":   "=o.o=",
        "blink":   "=-.-=",
        "error":   "=x.x=",
        "sleep":   "=u.u=",
        "wow":     "=O.O=",
        "sad":     "=T.T=",
        "proud":   "=^v^=",
        "box":     ("/\\_/\\", "( o.o )", " > ^ <"),
    }
    SPIN = SPINNER_FRAMES
    tty = False
    enabled = True
    face = "idle"

    # -- маленькие формы --------------------------------------------------- #
    @classmethod
    def line(cls, face: Optional[str] = None) -> str:
        return cls.FACES.get(face or cls.face, cls.FACES["idle"])

    @classmethod
    def say(cls, text: str = "", face: Optional[str] = None) -> None:
        """Реакция маскота на событие агента (одна строка после ответа)."""
        if not cls.tty:
            return
        f = face or cls.face
        color = C.RED if f in ("error", "sad") else (C.CYAN if f in ("think", "works") else C.MAGENTA)
        msg = _clip(text, max(20, _win_size()[0] - 20)) if text else ""
        sys.stdout.write(f"  {color}{cls.line(f)}{C.RESET} {C.GREY}{msg}{C.RESET}\n")
        sys.stdout.flush()

    @classmethod
    def box(cls, width: int = 0) -> str:
        top, face, paws = cls.FACES["box"]
        return "\n".join([f"  {C.BMAGENTA}{top}{C.RESET}",
                          f"  {C.BMAGENTA}{face}{C.RESET}",
                          f"  {C.BMAGENTA}{paws}{C.RESET}"])

    # -- кадры для анимаций ------------------------------------------------ #
    @staticmethod
    def chase_frame(pos: int, width: int) -> str:
        """Кот бежит по строке (pos 0..2*width): туда, затем обратно."""
        width = max(10, width)
        if pos < width:
            idx, trail_side, run = pos, -1, "happy"
        else:
            idx, trail_side, run = 2 * width - pos, +1, "wink"
        trail = " " * max(0, width - idx) if trail_side < 0 else ""
        right = "" if trail_side < 0 else " " * max(0, width - idx)
        dots = "·" * min(idx, 6)
        if trail_side < 0:
            return f"{trail}{dots}{C.BMAGENTA}{Mascot.line(run)}{C.RESET}{right}"
        return f"{trail}{C.BMAGENTA}{Mascot.line(run)}{C.RESET}{dots}{right}"

    @staticmethod
    def jump_frame(stage: int) -> str:
        """Прыжок: присед → вверх → вниз, с искрами."""
        seq = [
            ("  ᵕᵕ ", "=.-.="),
            (" ✧   ", "=^o^="),
            ("   ✧ ", "=^v^="),
            (" ᵕᵕ  ", "=^.^="),
        ]
        sparks, art = seq[stage % len(seq)]
        return f"{C.BYELLOW}{sparks}{C.RESET}{C.BMAGENTA}{art}{C.RESET}"

    @staticmethod
    def wave_frame(step: int) -> str:
        """Машет лапой: \\ / \\ /"""
        paws = ("\\", "/")
        phase = paws[step % 2]
        body = Mascot.line("happy" if step % 4 < 2 else "wink")
        return f"  {C.BCYAN}{phase}{C.RESET} {C.BMAGENTA}{body}{C.RESET}"

    @staticmethod
    def sleep_frame(step: int) -> str:
        """Спит и «дышит» (z летают)."""
        zs = ["   ", "  z", " zz", "zzz"][step % 4]
        return f"  {C.BMAGENTA}{Mascot.line('sleep')}{C.RESET} {C.GREY}{zs}{C.RESET}"

    # -- анимации ---------------------------------------------------------- #
    @staticmethod
    def _frame(line: Optional[str], seconds: float) -> None:
        """Рисует кадр в текущей строке (перезапись + обрезка остатка)."""
        if line is None:
            sys.stdout.write("\r" + " " * max(1, _win_size()[0] - 1) + "\r")
        else:
            body = line + C.RESET
            pad = max(0, _win_size()[0] - 1 - _disp_width(strip_ansi(body)))
            sys.stdout.write("\r" + body + " " * pad + "\r")
        sys.stdout.flush()
        if seconds:
            time.sleep(seconds)

    @classmethod
    def _play(cls, render: Callable[[int], str], frames: int, delay: float,
              final: str) -> None:
        try:
            for i in range(frames):
                cls._frame(render(i), delay)
        except KeyboardInterrupt:
            pass
        cls._frame(final, 0)

    @classmethod
    def intro(cls) -> None:
        """«Нейро на связи» печатается по буквам + подмигивание."""
        text = f"{MASCOT_NAME} на связи"
        if not cls.tty:
            print(f"  =^.^= {text}")
            return

        def render(i: int) -> str:
            shown = text[: min(i, len(text))]
            cursor = "▌" if i < len(text) else ""
            face = "wink" if i >= len(text) else "idle"
            return (f"  {C.BMAGENTA}{cls.line(face)}{C.RESET} " +
                    _tint(shown, C.BOLD) + _tint(cursor, C.MAGENTA))

        cls._play(render, len(text) + 8, 0.035,
                  f"  {C.BMAGENTA}{cls.line('happy')}{C.RESET} {C.BOLD}{text}{C.RESET}")

    @classmethod
    def wave(cls, rounds: int = 2) -> None:
        """Машет лапой (в ответ на приветствие пользователя)."""
        if not cls.tty:
            return
        cls._play(lambda i: cls.wave_frame(i),
                  rounds * 4, 0.12,
                  f"  {C.BMAGENTA}{cls.line('happy')}{C.RESET}")

    @classmethod
    def run(cls) -> None:
        """Кот пробегает по экрану — во время загрузки llama.cpp / модели."""
        if not cls.tty:
            return
        width = max(20, min(56, _win_size()[0] - 8))
        cls._play(lambda i: cls.chase_frame(i, width),
                  width * 2 + 1, 0.012,
                  f"  {C.BMAGENTA}{cls.line('idle')}{C.RESET}")

    @classmethod
    def jump(cls, face: str = "happy") -> None:
        """Радостный прыжок с искрами — после успешной задачи."""
        if not cls.tty:
            return
        cls._play(lambda i: cls.jump_frame(i), 8, 0.09,
                  f"  {C.BMAGENTA}{cls.line(face)}{C.RESET}")

    @classmethod
    def sleep(cls, seconds: float = 1.2) -> None:
        """«Спит» — когда что-то долго не происходит."""
        if not cls.tty:
            return
        cls._play(lambda i: cls.sleep_frame(i), int(seconds / 0.25), 0.25,
                  f"  {C.BMAGENTA}{cls.line('sleep')}{C.RESET}")

    @classmethod
    def draw_help(cls) -> None:
        print(f"  {C.BMAGENTA}{cls.line('idle')}{C.RESET} {C.BOLD}{MASCOT_NAME}{C.RESET}"
              f" — кот-помощник: читает, пишет и правит файлы, ищет код и запускает команды.")


class MascotAnim:
    """Живая строка-анимация маскота в одной строке терминала."""

    def __init__(self, label: str = "думаю", working: bool = True,
                 force_tty: bool = False) -> None:
        self.label = label
        self.working = working
        self.stop_ev = threading.Event()
        # анимация не рисуется, если терминал не интерактивный
        self.tty = bool(force_tty or (_live_terminal() and Mascot.enabled))
        self.live = bool(_live_terminal()) and Mascot.enabled
        self.t0 = time.time()
        self.th = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> "MascotAnim":
        if self.live:
            self.th.start()
        elif self.working and not self.tty:
            print(f"  {self.label}…", flush=True)
        return self

    def _frame(self, i: int) -> Tuple[str, str]:
        # 12 тактов: думает → барабанит лапами → мигает → думает → мигает
        beat = i % 12
        spin = Mascot.SPIN[i % len(Mascot.SPIN)]
        if not self.working:
            if beat == 5 or beat == 11:
                return Mascot.line("blink"), spin
            return Mascot.line("idle"), Mascot.SPIN[(i * 2) % len(Mascot.SPIN)]
        if beat < 2:
            return Mascot.line("think"), spin
        if beat < 5:
            return Mascot.line("works"), spin
        if beat == 5:
            return Mascot.line("blink"), spin
        if beat < 9:
            return Mascot.line("think"), Mascot.SPIN[(i * 3) % len(Mascot.SPIN)]
        if beat == 10:
            return Mascot.line("blink"), spin
        return Mascot.line("idle"), spin

    def _build_line(self, i: int) -> str:
        art, spin = self._frame(i)
        dt = time.time() - self.t0
        return (f"  {C.BMAGENTA}{art}{C.RESET} {C.BCYAN}{spin}{C.RESET} "
                f"{self.label}… {C.GREY}{dt:4.0f}с{C.RESET}")

    def _run(self) -> None:
        i = 0
        while not self.stop_ev.is_set():
            if self.tty:
                try:
                    self._frame_draw(self._build_line(i))
                except Exception:
                    return
            i += 1
            time.sleep(0.09)

    def _frame_draw(self, line: str) -> None:
        pad = max(0, _win_size()[0] - 1 - _disp_width(strip_ansi(line)))
        sys.stdout.write("\r" + line + C.RESET + " " * pad)
        sys.stdout.flush()

    def stop(self) -> None:
        if self.stop_ev.is_set():
            return
        self.stop_ev.set()
        if self.th.is_alive():
            self.th.join(timeout=1)
        if self.tty:
            try:
                sys.stdout.write("\r" + " " * max(10, _win_size()[0] - 1) + "\r")
                sys.stdout.flush()
            except Exception:
                pass

    def __exit__(self, *_exc: Any) -> None:
        self.stop()

    def celebrate(self, face: str = "happy", text: str = "") -> None:
        """Короткая реакция маскота вместо строки анимации."""
        self.stop()
        if not text:
            return
        if self.tty:
            sys.stdout.write(f"  {C.BMAGENTA}{Mascot.line(face)}{C.RESET} "
                             f"{C.GREY}{_clip(text, 100)}{C.RESET}\n")
        else:
            sys.stdout.write(f"  {text}\n")
        sys.stdout.flush()


# --------------------------------------------------------------------------- #
#                ДОМАШНЯЯ ДИРЕКТОРИЯ: ВСЁ ПРОГРАММЫ В ОДНОМ МЕСТЕ
# --------------------------------------------------------------------------- #

HOME_POINTER = Path(__file__).resolve().parent / ".local-agent-home.json"
HOME_DIR_MARKERS = ("llama.cpp", "models", "tmp", "undo", "config.json", "venv")
STAI_SHIM_MARK = ".stai-shim.json"
VENV_EXE_WIN = ("Scripts", "python.exe")
VENV_EXE_POSIX = ("bin", "python")
ALIAS_UNIX_NAMES = ("stai", "stai.sh")


def resolve_home() -> Optional[Path]:
    """Папка из окружения или запомненного выбора. None => первый запуск."""
    env = os.environ.get("LOCAL_AGENT_HOME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    try:
        data = json.loads(HOME_POINTER.read_text(encoding="utf-8"))
        saved = str((data or {}).get("home") or "").strip()
        if saved:
            return Path(saved).expanduser().resolve()
    except Exception:
        pass
    return None


def default_home() -> Path:
    """Куда ставим «по умолчанию» — стандартный каталог приложения."""
    if IS_WINDOWS:
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    if IS_MAC:
        return Path.home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def portable_home() -> Path:
    """Домашняя папка рядом со скриптом (режим «на флешке»)."""
    return Path(__file__).resolve().parent / APP_NAME


def candidate_homes() -> List[Path]:
    """Разумные варианты папки для режима «сразу Enter»."""
    out: List[Path] = []
    if IS_WINDOWS:
        # Windows ограничивает путь для длинных имён > 260 символов, папка в профиле
        # обычно заметно короче — поэтому на первом месте %APPDATA%\local-agent
        out.append(default_home())
        out.append(portable_home())
        if Path.home().is_dir():
            out.append(Path.home() / APP_NAME)
    else:
        out.extend([default_home(), portable_home()])
    uniq: List[Path] = []
    for p in out:
        try:
            rp = p.resolve()
        except Exception:
            continue
        if rp not in uniq:
            uniq.append(rp)
    return uniq


def home_paths(home: Optional[Path] = None) -> Dict[str, Path]:
    """ЕДИНАЯ папка программы: модели, бинарники, конфиг, логи, откаты, venv."""
    home = Path(home).expanduser().resolve() if home else resolve_home()
    if home is None:
        home = default_home()
    return {
        "home": home,
        "bin": home / "llama.cpp",
        "models": home / "models",
        "tmp": home / "tmp",
        "workspace": home / "workspace",
        "config": home / "config.json",
        "undo": home / "undo",
        "last_model": home / "last_model.txt",
        "logs": home / "logs",
        "venv": home / "venv",
        "state": home / "state.json",
    }


def ensure_dirs(P: Dict[str, Path]) -> None:
    for key in ("home", "bin", "models", "tmp", "workspace", "undo", "logs"):
        try:
            P[key].mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Не удалось создать папку {P[key]}: {e}") from e


def save_home_pointer(home: Path) -> bool:
    """Запоминает выбор каталога рядом со скриптом, чтобы следующий запуск был молчаливым."""
    try:
        HOME_POINTER.parent.mkdir(parents=True, exist_ok=True)
        prev: Dict[str, Any] = {}
        if HOME_POINTER.is_file():
            try:
                prev = json.loads(HOME_POINTER.read_text(encoding="utf-8")) or {}
            except Exception:
                prev = {}
        prev.update({"home": str(home), "updated": datetime.now().isoformat(timespec="seconds")})
        tmp = HOME_POINTER.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(prev, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, HOME_POINTER)
        return True
    except Exception as e:
        warn(f"Не удалось сохранить выбор каталога ({HOME_POINTER}): {e}")
        return False


def load_state(P: Dict[str, Path]) -> Dict[str, Any]:
    try:
        data = json.loads(P["state"].read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_state(P: Dict[str, Path], state: Dict[str, Any]) -> None:
    try:
        ensure_dirs(P)
        tmp = P["state"].with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, P["state"])
    except Exception as e:
        warn(f"Не удалось сохранить состояние: {e}")


def dir_is_ours(home: Path) -> bool:
    if not home.is_dir():
        return False
    try:
        return any((home / m).exists() for m in HOME_DIR_MARKERS)
    except OSError:
        return False


def dir_is_emptyish(home: Path) -> bool:
    if not home.exists():
        return True
    try:
        return not any(home.iterdir())
    except OSError:
        return False


def dir_writable(home: Path) -> bool:
    probe = None
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / f".write-test-{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        try:
            if probe is not None:
                probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def _canon(p: Path) -> Tuple[str, ...]:
    return tuple(x.lower() for x in p.resolve().parts)


def _is_inside(child: Path, parent: Path) -> bool:
    c, p = _canon(child), _canon(parent)
    return len(c) >= len(p) and c[: len(p)] == p


def _copy_tree(src: Path, dst: Path, log: Optional[Callable[[str], None]] = None) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for item in src.iterdir():
        target = dst / item.name
        try:
            if item.is_dir():
                n += _copy_tree(item, target, log)
            else:
                shutil.copy2(item, target)
                n += 1
        except OSError as e:
            if log:
                log(f"пропуск {item}: {e}")
    return n


def migrate_home(src: Path, dst: Path) -> int:
    """Переносит содержимое старой папки программы в новую (модели не копируются дважды)."""
    if not src.exists():
        return 0
    moved = 0
    for item in src.iterdir():
        target = dst / item.name
        try:
            if item.is_dir():
                if target.exists():
                    moved += _copy_tree(item, target, warn)
                else:
                    try:
                        shutil.move(str(item), str(target))
                        moved += 1
                    except OSError:
                        moved += _copy_tree(item, target, warn)
            else:
                if target.exists():
                    continue
                shutil.copy2(item, target)
                moved += 1
        except OSError as e:
            warn(f"Не удалось перенести {item.name}: {e}")
    return moved


def setup_home_interactive(path_arg: str = "") -> Path:
    """Первичная настройка: выбрать ОДНУ папку программы и запомнить её."""
    rule("Папка программы")
    print(f"  {C.GREY}Всё будет храниться в одном каталоге: llama.cpp, модели, конфиг,")
    print(f"  логи, журнал откатов, venv и рабочая папка агента.{C.RESET}")

    if path_arg:
        home = Path(path_arg).expanduser()
        if not dir_writable(home):
            err(f"В папку нельзя писать: {home}")
            raise SystemExit(1)
        save_home_pointer(home)
        ok(f"Папка программы: {home.resolve()}")
        return home.resolve()

    suggested = candidate_homes()
    for i, cand in enumerate(suggested, 1):
        mark = ""
        if dir_is_ours(cand):
            mark = f"  {C.GREEN}← уже настроена{C.RESET}"
        elif not dir_is_emptyish(cand):
            mark = f"  {C.GREY}(папка не пуста){C.RESET}"
        print(f"  {C.CYAN}{i}{C.RESET}) {cand}{mark}")
    print(f"  {C.CYAN}{len(suggested) + 1}{C.RESET}) указать свой путь")

    while True:
        v = ask("Выберите папку программы", "1")
        try:
            n = int(v)
        except ValueError:
            n = -1
        if 1 <= n <= len(suggested):
            home = suggested[n - 1]
        elif n == len(suggested) + 1:
            raw = ask("Полный путь к папке")
            if not raw:
                err("Пустой путь.")
                continue
            home = Path(raw).expanduser()
        else:
            err("Введите номер из списка.")
            continue

        home = home.resolve()
        # сначала проверяем, что папка вообще доступна на запись,
        # и только потом спрашиваем об уже существующих данных
        if not dir_writable(home):
            err(f"В папку нельзя писать: {home}. Выберите другую.")
            continue
        if dir_is_ours(home) and not ask_yes_no(
                f"В {home} уже есть данные программы — использовать их?", True):
            continue
        break

    # перенос из старой папки по умолчанию, если она есть и это другая папка
    legacy = default_home().resolve()
    if legacy != home and dir_is_ours(legacy):
        if ask_yes_no(f"Перенести данные из {legacy}?", True):
            n = migrate_home(legacy, home)
            if n:
                ok(f"Перенесено объектов: {n}")

    ensure_dirs(home_paths(home))
    save_home_pointer(home)
    ok(f"Папка программы: {home}")
    return home


# --------------------------------------------------------------------------- #
#           ЗАПУСК ОДНОЙ КОМАНДОЙ: ЗАВИСИМОСТИ, VENV И АЛИАС `stai`
# --------------------------------------------------------------------------- #

CORE_DEPS: List[Tuple[str, str]] = []   # (модуль, pip-имя); ядро работает на stdlib
OPTIONAL_DEPS: List[Tuple[str, str]] = [
    ("psutil", "psutil"),               # детектор железа в _hw_probe.py
    ("certifi", "certifi"),             # надёжный HTTPS на macOS/Windows
]
SKIP_BOOTSTRAP_ENV = "LOCAL_AGENT_NO_BOOTSTRAP"
LAUNCHERS = (".cmd", ".bat", ".sh", ".ps1", ".py")
REQUIRED_PY = (3, 8)


def module_available(name: str) -> bool:
    try:
        import importlib.util
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def missing_modules(deps: Sequence[Tuple[str, str]]) -> List[Tuple[str, str]]:
    return [(m, p) for m, p in deps if not module_available(m)]


def venv_python(P: Dict[str, Path]) -> Optional[Path]:
    for parts in (VENV_EXE_WIN, VENV_EXE_POSIX):
        cand = P["venv"].joinpath(*parts)
        if cand.is_file():
            return cand
    return None


def running_in_venv(P: Dict[str, Path]) -> bool:
    vp = venv_python(P)
    if not vp:
        return False
    try:
        return Path(sys.executable).resolve() == vp.resolve()
    except Exception:
        return False


def pip_install(py: Path, packages: Sequence[str], timeout: int = 600) -> Tuple[bool, str]:
    if not packages:
        return True, ""
    cmd = [str(py), "-m", "pip", "install", "--disable-pip-version-check",
           "--no-input", *packages]
    note("Устанавливаю: " + " ".join(packages))
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             stdin=subprocess.DEVNULL)
    except Exception as e:
        return False, str(e)
    if res.returncode != 0:
        tail = strip_ansi((res.stderr or res.stdout or "")).strip().splitlines()[-12:]
        return False, "\n".join(tail)
    return True, ""


def create_venv(P: Dict[str, Path]) -> Optional[Path]:
    """Создаёт venv в папке программы. Возвращает python из него или None."""
    note(f"Создаю виртуальное окружение в {P['venv']}…")
    P["venv"].parent.mkdir(parents=True, exist_ok=True)
    try:
        res = subprocess.run([sys.executable, "-m", "venv", str(P["venv"])],
                             capture_output=True, text=True, timeout=300,
                             stdin=subprocess.DEVNULL)
        if res.returncode != 0:
            warn("venv не создался: " + strip_ansi((res.stderr or "").strip().splitlines()[-1]
                                                   if res.stderr.strip() else "неизвестная ошибка"))
            return None
    except Exception as e:
        warn(f"venv не создался: {e}")
        return None
    py = venv_python(P)
    if not py:
        warn("venv создан, но python внутри не найден.")
        return None
    # в venv без pip доустанавливаем его
    try:
        chk = subprocess.run([str(py), "-m", "pip", "--version"], capture_output=True,
                             text=True, timeout=120, stdin=subprocess.DEVNULL)
        if chk.returncode != 0:
            subprocess.run([str(py), "-m", "ensurepip", "--upgrade"], capture_output=True,
                           text=True, timeout=180, stdin=subprocess.DEVNULL)
    except Exception:
        pass
    ok(f"Виртуальное окружение готово: {py}")
    return py


def package_initialized(P: Dict[str, Path]) -> bool:
    st = load_state(P)
    return bool(st.get("initialized")) and bool(st.get("alias"))


def bootstrap(P: Dict[str, Path], auto: bool = True) -> None:
    """Первый запуск: поставить зависимости в venv, настроить окружение, прописать `stai`.

    Вызывается один раз. Если создаётся venv, скрипт перезапускает сам себя
    интерпретатором из venv (через env-флаг — без зацикливания), поэтому все
    последующие запуски требуют только одной команды: stai.
    """
    rule("Первичная настройка")
    if sys.version_info[:2] < REQUIRED_PY:
        warn(f"Рекомендуется Python {REQUIRED_PY[0]}.{REQUIRED_PY[1]}+; запущен "
             f"{sys.version_info.major}.{sys.version_info.minor}. Продолжаю на stdlib.")

    state = load_state(P)
    state["main_script"] = str(Path(__file__).resolve())
    save_state(P, state)

    # venv уже есть, а мы не в нём — сразу перезапускаемся в нём
    existing_venv = venv_python(P)
    if existing_venv and not running_in_venv(P):
        relaunch_in_venv(P, existing_venv)

    need_core = missing_modules(CORE_DEPS)
    need_opt = missing_modules(OPTIONAL_DEPS)

    # 1) venv: создаём, только если зависимостей реально не хватает
    if not running_in_venv(P):
        if need_core or (need_opt and auto):
            note("Ставлю зависимости один раз, чтобы запуск шёл одной командой…")
            venv_py = create_venv(P)
            if venv_py:
                relaunch_in_venv(P, venv_py)
            else:
                warn("Работаю на системном Python (без venv).")
        else:
            ok("Внешние зависимости не требуются — работаю на стандартной библиотеке.")

    got_venv = running_in_venv(P)
    if got_venv:
        miss = missing_modules(OPTIONAL_DEPS)
        if miss:
            good, tail = pip_install(Path(sys.executable), [p for _, p in miss])
            if good:
                ok("Дополнительные пакеты установлены: " + ", ".join(p for _, p in miss))
            else:
                warn(f"Часть пакетов не установилась (не критично):\n{tail}")
        else:
            ok("Зависимости на месте.")

    # 2) команда stai
    alias_ok = False
    try:
        alias_ok = register_alias(P)
    except Exception as e:
        warn(f"Не удалось прописать команду stai: {e}")

    # 3) окружение оболочки (резервный alias в профиле PowerShell + LOCAL_AGENT_HOME)
    try:
        setup_shell_env(P)
    except Exception as e:
        warn(f"Не удалось настроить окружение оболочки: {e}")

    st = load_state(P)
    st.update({
        "initialized": True,
        "python": sys.executable,
        "venv": str(P["venv"]) if got_venv else "",
        "main_script": str(Path(__file__).resolve()),
        "alias": bool(alias_ok or st.get("alias")),
        "initialized_at": datetime.now().isoformat(timespec="seconds"),
    })
    save_state(P, st)

    print()
    if alias_ok:
        ok("Настройка завершена. Запуск из любого каталога: " + C.BOLD + "stai" + C.RESET)
        print(f"  {C.GREY}Если в текущем терминале команда не видна — откройте новое окно.{C.RESET}")
    else:
        warn("Команда stai не прописана. Запасной запуск одной командой:")
        print(f"  {C.GREY}{' '.join(launch_cmd())}{C.RESET}")


def relaunch_in_venv(P: Dict[str, Path], venv_py: Path) -> None:
    """Перезапуск того же скрипта интерпретатором из venv (без зацикливания)."""
    target = Path(__file__).resolve()
    # --home передаём только если папку задали явно/запомнили, иначе
    # перезапуск мог бы принять временное значение за пользовательский выбор
    args = [str(venv_py), str(target)]
    if os.environ.get("LOCAL_AGENT_HOME", "").strip():
        args += ["--home", str(P["home"])]
    args += sys.argv[1:]
    env = dict(os.environ)
    env["LOCAL_AGENT_BOOTSTRAPPED"] = "1"
    env["LOCAL_AGENT_HOME"] = str(P["home"])
    note(f"Перезапускаю в venv: {venv_py}")
    try:
        res = subprocess.run(args, env=env)
        raise SystemExit(res.returncode)
    except OSError as e:
        warn(f"Не удалось перезапуститься в venv ({e}) — продолжаю на {sys.executable}")


def launcher_script(P: Dict[str, Path]) -> Path:
    """Точка входа для алиаса: <домашняя папка>/scripts/stai.py."""
    return P["home"] / "scripts" / "stai.py"


def launch_cmd() -> List[str]:
    """Резервная команда на случай, если алиас не прописался."""
    if IS_WINDOWS:
        return [sys.executable, str(Path(__file__).resolve())]
    py = sys.executable or "python3"
    return [py, str(Path(__file__).resolve())]


def _resolve_python_for_alias(P: Dict[str, Path], recorded: str = "") -> str:
    vp = venv_python(P)
    if vp:
        return str(vp)
    for cand in (recorded, sys.executable, shutil.which("python"), shutil.which("python3"),
                 shutil.which("py")):
        if cand and Path(cand).exists():
            return str(cand)
    return sys.executable or "python3"


def install_launchers(P: Dict[str, Path], python_exe: str) -> List[Path]:
    """Создаёт файлы-шорткаты (stai.cmd / stai.ps1 / stai.sh) в папке программы."""
    scripts = P["home"] / "scripts"
    try:
        scripts.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise RuntimeError(f"Нет доступа к {scripts}: {e}") from e
    launcher = scripts / "stai.py"
    # единый файл: launcher — это обёртка, которая запускает этот же скрипт,
    # поэтому stai продолжает работать, даже если исходник переместить
    try:
        shutil.copy2(Path(__file__).resolve(), launcher)
    except OSError as e:
        raise RuntimeError(f"Не удалось создать launcher {launcher}: {e}") from e
    made: List[Path] = []
    if IS_WINDOWS:
        cmd = P["home"] / "stai.cmd"
        # только ASCII: cmd.exe в OEM-кодировке неверно читает UTF-8 кириллицу
        cmd.write_text(
            "@echo off\r\n"
            "rem local-agent launcher (autogenerated)\r\n"
            f'"{python_exe}" "{launcher}" %*\r\n',
            encoding="ascii", newline="")
        made.append(cmd)
        ps1 = P["home"] / "stai.ps1"
        ps1.write_text(
            "# local-agent launcher (autogenerated)\n"
            f'& "{python_exe}" "{launcher}" @args\n'
            "exit $LASTEXITCODE\n",
            encoding="utf-8-sig")
        made.append(ps1)
    else:
        sh = P["home"] / "stai.sh"
        sh.write_text(
            "#!/bin/sh\n# local-agent launcher (autogenerated)\n"
            f'exec "{python_exe}" "{launcher}" "$@"\n',
            encoding="utf-8")
        try:
            sh.chmod(0o755)
        except OSError:
            pass
        made.append(sh)
    st = load_state(P)
    st["main_script"] = str(Path(__file__).resolve())
    save_state(P, st)
    return made


def user_bin_dir() -> Path:
    if IS_WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "Microsoft" / "WindowsApps"
    return Path.home() / ".local" / "bin"


def _dir_in_path(d: Path) -> bool:
    want = str(d).lower().rstrip("\\/")
    for item in os.environ.get("PATH", "").split(os.pathsep):
        item = item.strip().strip('"')
        if item and str(Path(item)).lower().rstrip("\\/") == want:
            return True
    return False


def _read_user_path_raw() -> str:
    """Сырое значение HKCU\\Environment\\Path (с %USERPROFILE%, без раскрытия)."""
    if not IS_WINDOWS:
        return ""
    try:
        import winreg  # type: ignore[import-not-found]
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
            val, _ = winreg.QueryValueEx(k, "Path")
            return str(val or "")
    except Exception:
        return ""


def ensure_dir_in_user_path(d: Path) -> bool:
    """Добавляет папку в пользовательский PATH (HKCU), если её там нет.

    Пишется сырое REG_EXPAND_SZ-значение, поэтому %USERPROFILE% остаётся
    нераскрытым. Возвращает True, если изменение было сделано (нужен новый терминал).
    """
    if not IS_WINDOWS:
        return False
    if _dir_in_path(d):
        return False
    try:
        import winreg  # type: ignore[import-not-found]
        want = str(d)
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                            winreg.KEY_READ | winreg.KEY_WRITE) as k:
            try:
                raw, kind = winreg.QueryValueEx(k, "Path")
                raw = str(raw or "")
            except FileNotFoundError:
                raw, kind = "", winreg.REG_EXPAND_SZ
            current = [p for p in raw.split(";") if p.strip()]
            low = {p.strip().lower() for p in current}
            if want.lower() in low:
                return False
            current.append(want)
            new = ";".join(current)
            winreg.SetValueEx(k, "Path", 0, kind or winreg.REG_EXPAND_SZ, new)
        os.environ["PATH"] = os.environ.get("PATH", "") + os.pathsep + want
        return True
    except Exception as e:
        warn(f"Не удалось добавить {d} в PATH пользователя: {e}")
        return False


def ensure_plugin_dir() -> Optional[Path]:
    """Своя папка-«плагин» пути: всегда доступна для записи и не зависит от прав на WindowsApps."""
    try:
        if IS_WINDOWS:
            base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        else:
            base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
        d = (Path(base) / APP_NAME / "bin").resolve()
        d.mkdir(parents=True, exist_ok=True)
        return d
    except OSError:
        return None


def _safe_unlink(p: Path) -> None:
    try:
        if p.is_symlink() or p.is_file():
            p.unlink()
    except OSError:
        pass


def shim_kind(target: Path) -> str:
    """Что лежит в папке PATH под именем stai: наш shim, чужой файл или ничего."""
    if not target.exists() and not target.is_symlink():
        return "free"
    try:
        if IS_WINDOWS and target.suffix.lower() == ".cmd":
            data = target.read_bytes()[:400].decode("utf-8", errors="replace")
            if "local-agent launcher" in data:
                return "ours"
        elif not IS_WINDOWS:
            data = target.read_bytes()[:400].decode("utf-8", errors="replace")
            if "local-agent launcher" in data:
                return "ours"
        mark = target.parent / STAI_SHIM_MARK
        if mark.is_file():
            try:
                data = json.loads(mark.read_text(encoding="utf-8"))
                if str(data.get("shim", "")).lower() == str(target).lower():
                    return "ours"
            except Exception:
                pass
        return "foreign"
    except OSError:
        return "foreign"


def register_alias(P: Dict[str, Path], quiet: bool = False) -> bool:
    """Прописывает команду `stai` в папку, которая уже есть в PATH.

    Windows: %LOCALAPPDATA%\\Microsoft\\WindowsApps — единственная стандартная
    пользовательская папка в PATH (чужой файл там не перезаписывается).
    POSIX: ~/.local/bin.
    """
    state = load_state(P)
    python_exe = _resolve_python_for_alias(P, str(state.get("python") or ""))
    try:
        install_launchers(P, python_exe)
    except OSError as e:
        if not quiet:
            warn(f"Не удалось создать launcher: {e}")
        return False

    target_dir = user_bin_dir()
    src = P["home"] / ("stai.cmd" if IS_WINDOWS else "stai.sh")
    if not src.is_file():
        if not quiet:
            warn("launcher не создан — пропускаю установку команды.")
        return False

    path_added = False
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        if not quiet:
            warn(f"Нет доступа к {target_dir}: {e}")
        target_dir = None  # type: ignore[assignment]
    if target_dir is not None and not _dir_in_path(target_dir):
        if ensure_dir_in_user_path(target_dir):
            path_added = True
            note(f"{target_dir} добавлен в PATH пользователя.")

    if target_dir is not None:
        dest = target_dir / src.name
        kind = shim_kind(dest)
        if kind == "foreign":
            if not quiet:
                warn(f"В {dest} уже лежит посторонний файл — не трогаю его.")
        else:
            try:
                if kind == "ours":
                    _safe_unlink(dest)
                if IS_WINDOWS:
                    shutil.copy2(src, dest)
                else:
                    try:
                        os.symlink(str(src), str(dest))
                    except OSError:
                        shutil.copy2(src, dest)
                        try:
                            dest.chmod(0o755)
                        except OSError:
                            pass
                try:
                    (target_dir / STAI_SHIM_MARK).write_text(
                        json.dumps({"shim": str(dest),
                                    "source": str(Path(__file__).resolve()),
                                    "home": str(P["home"]),
                                    "updated": datetime.now().isoformat(timespec="seconds")},
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8")
                except OSError:
                    pass
                st = load_state(P)
                st.update({"alias": True, "alias_kind": "path", "alias_path": str(dest),
                           "python": python_exe})
                save_state(P, st)
                if not quiet:
                    ok(f"Команда {C.BOLD}stai{C.RESET} прописана: {dest}"
                       + ("  (понадобится новое окно терминала)" if path_added else ""))
                return True
            except OSError as e:
                if not quiet:
                    warn(f"Не удалось поставить {dest}: {e}")

    # запасной путь: своя папка, которая точно доступна на запись
    plugin_dir = ensure_plugin_dir()
    dest2: Optional[Path] = None
    if plugin_dir is not None:
        dest2 = plugin_dir / src.name
        try:
            plugin_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest2)
        except OSError as e:
            if not quiet:
                warn(f"Не удалось поставить {dest2}: {e}")
        else:
            added = ensure_dir_in_user_path(plugin_dir)
            st = load_state(P)
            st.update({"alias": True, "alias_kind": "plugin", "alias_path": str(dest2),
                       "python": python_exe})
            save_state(P, st)
            if not quiet:
                ok(f"Команда {C.BOLD}stai{C.RESET} прописана: {dest2}")
                if added:
                    note(f"{plugin_dir} добавлен в PATH — откройте новое окно терминала.")
            return True

    if not quiet:
        warn("Команду stai автоматически прописать не удалось — смотрите запасной вариант ниже.")
    return False


PS_PROFILE_START = "# >>> local-agent (stai) >>>"
PS_PROFILE_END = "# <<< local-agent (stai) <<<"


def ps_profile_paths() -> List[Path]:
    if not IS_WINDOWS:
        return []
    try:
        res = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                              "$PROFILE.CurrentUserAllHosts"],
                             capture_output=True, text=True, timeout=20,
                             stdin=subprocess.DEVNULL)
        out = (res.stdout or "").strip().splitlines()
        if out and out[-1].strip():
            return [Path(out[-1].strip())]
    except Exception:
        pass
    return []


def setup_shell_env(P: Dict[str, Path]) -> None:
    """Резервный (и дополнительный) способ: alias-функция в профиле PowerShell.

    Профиль PowerShell при старте не исполняется, если политика запрещает скрипты
    (в этой системе так и есть), поэтому основной путь — shim в WindowsApps,
    который PowerShell находит в PATH как `stai.cmd`.
    """
    if not IS_WINDOWS:
        return
    if not (P["home"] / "stai.cmd").is_file():
        try:
            install_launchers(P, _resolve_python_for_alias(P))
        except OSError:
            return

    cmd_path = P["home"] / "stai.cmd"

    state0 = load_state(P)
    if not state0.get("main_script"):
        state0["main_script"] = str(Path(__file__).resolve())
        save_state(P, state0)

    block = "\n".join([
        PS_PROFILE_START,
        f"$env:LOCAL_AGENT_HOME = '{P['home']}'",
        f"function global:stai {{ & '{cmd_path}' @args }}",
        PS_PROFILE_END,
        "",
    ])
    written = False
    for prof in ps_profile_paths():
        try:
            prof.parent.mkdir(parents=True, exist_ok=True)
            text = prof.read_text(encoding="utf-8", errors="replace") if prof.is_file() else ""
            if PS_PROFILE_START in text:
                head = text.split(PS_PROFILE_START)[0].rstrip()
                tail = ""
                if PS_PROFILE_END in text:
                    tail = text.split(PS_PROFILE_END, 1)[1].lstrip("\r\n")
                text = head + "\n" + block + tail
            else:
                text = (text.rstrip() + "\n\n" if text.strip() else "") + block
            prof.write_text(text, encoding="utf-8")
            written = True
            note(f"Профиль PowerShell обновлён: {prof}")
        except OSError as e:
            warn(f"Не удалось обновить профиль {prof}: {e}")

    # копия .cmd рядом со скриптом — работает как fallback в PATH-папках
    try:
        twin = Path(__file__).resolve().parent / "stai.cmd"
        if cmd_path.is_file() and twin != cmd_path:
            shutil.copy2(cmd_path, twin)
    except OSError:
        pass

    state = load_state(P)
    state["alias"] = state.get("alias") or written
    if written:
        state["alias_kind"] = "powershell-profile"
    save_state(P, state)


DEFAULT_CONFIG: Dict[str, Any] = {
    "ctx": 0,           # 0 => llama-server возьмёт из модели (обычно 4096-32768)
    "threads": 0,       # 0 => авто (-1 у llama-server)
    "gpu_layers": -1,   # -1 = авто; 0 = только CPU
    "approve": True,    # спрашивать разрешение на команды оболочки/запись
    "flash_attn": "auto",
    "no_mmap": False,
    "hf_token": "",     # токен для gated-моделей (env HF_TOKEN тоже работает)
    "api_key": "",      # необязательный ключ для llama-server
    "extra_args": [],   # доп. аргументы llama-server
    "port": 0,          # 0 => свободный порт
}


def load_config(P: Dict[str, Path]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    cfg["extra_args"] = []
    try:
        if P["config"].is_file():
            data = json.loads(P["config"].read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for k, v in data.items():
                    if k in cfg:
                        cfg[k] = v
    except Exception as e:
        warn(f"Не удалось прочитать конфиг {P['config']}: {e}")
    return cfg


def save_config(P: Dict[str, Path], cfg: Dict[str, Any]) -> None:
    try:
        ensure_dirs(P)
        tmp = P["config"].with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, P["config"] if "config" in P else P["home"] / "config.json")
    except Exception as e:
        warn(f"Не удалось сохранить конфиг: {e}")


def fmt_bytes(n: float) -> str:
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if abs(n) < 1024 or unit == "ТБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ТБ"


def fmt_duration(sec: Optional[float]) -> str:
    if sec is None or sec != sec or sec < 0 or sec == float("inf"):
        return "--:--"
    sec = int(sec)
    if sec >= 3600:
        return f"{sec // 3600}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
    return f"{sec // 60:02d}:{sec % 60:02d}"


def human_time(ts: float) -> str:
    try:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "?"


class ProgressBar:
    """Живой прогресс-бар в одной строке терминала."""

    def __init__(self, label: str = "", width: int = 30) -> None:
        self.label = label
        self.width = width
        self.enabled = _live_terminal()
        self.start_ts = time.time()
        self._last = 0.0
        self._last_line_len = 0

    def update(self, done: int, total: int, speed: float = 0.0) -> None:
        if not self.enabled:
            now = time.time()
            if now - self._last < 1.0 and total and done < total:
                return
            self._last = now
            pct = f"{done / total * 100:5.1f}%" if total else "  ??? "
            print(f"  {self.label} {pct} {fmt_bytes(done)}" + (f"/{fmt_bytes(total)}" if total else ""), flush=True)
            return
        now = time.time()
        if total:
            frac = min(1.0, done / total)
            filled = int(self.width * frac)
            bar = "█" * filled + "░" * (self.width - filled)
            pct = f"{frac * 100:5.1f}%"
            speed_txt = f"{fmt_bytes(speed)}/с" if speed else "…"
            eta_txt = fmt_duration((total - done) / speed) if speed > 0 else "--:--"
            core = (f"{bar} {pct}  {fmt_bytes(done)}/{fmt_bytes(total)}"
                    f"  {speed_txt}  ETA {eta_txt}")
        else:
            frames = len(SPINNER_FRAMES)
            spin = SPINNER_FRAMES[int(now * 10) % frames]
            core = f"{spin} {fmt_bytes(done)}  {fmt_bytes(speed)}/с"
        if now - self._last < 0.05 and done != total:
            return
        self._last = now
        cols, _ = _win_size()
        line = f"  {self.label} {core}"
        line = _clip(line, cols - 1)
        pad = ""
        sys.stdout.write("\r" + line + pad)
        self._last_line_len = _disp_width(line)
        sys.stdout.flush()

    def finish(self, msg: str = "") -> None:
        if self.enabled:
            sys.stdout.write("\r" + " " * self._last_line_len + "\r")
            sys.stdout.flush()
        if msg:
            print(msg, flush=True)


def fmt_table(rows: List[Tuple[str, str, str]], headers: Tuple[str, str, str]) -> None:
    col2 = max([len(headers[1])] + [len(r[1]) for r in rows])
    print(C.BOLD + f" {headers[0]:<4} {headers[1]:<{col2}} {headers[2]}" + C.RESET)
    for a, b, c in rows:
        print(f" {C.CYAN}{a:<4}{C.RESET} {b:<{col2}} {C.GREY}{c}{C.RESET}")


def ask(prompt: str, default: str = "") -> str:
    try:
        suffix = f" [{default}]" if default else ""
        val = input(f"{C.BCYAN}{prompt}{suffix}›{C.RESET} ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        raise SystemExit(0)
    return val or default


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        v = ask(f"{prompt}", d).lower()
        if v in ("y", "yes", "д", "да", "1"):
            return True
        if v in ("n", "no", "н", "нет", "0"):
            return False


def choose(prompt: str, items: List[str], default: int = 1) -> int:
    """items — уже отформатированные строки; возвращает индекс 0..n-1."""
    for i, it in enumerate(items, 1):
        print(f"  {C.CYAN}{i:>2}{C.RESET}) {it}")
    while True:
        v = ask(prompt, str(default))
        try:
            n = int(v)
            if 1 <= n <= len(items):
                return n - 1
        except ValueError:
            pass
        err("Введите число из списка.")


class _Net:
    """Минимальный HTTP-хелпер на urllib (с ретраями на 5xx/502 от GitHub)."""

    @staticmethod
    def request(url: str, method: str = "GET", data: Optional[bytes] = None,
                headers: Optional[Dict[str, str]] = None, timeout: int = 60):
        h = {"User-Agent": USER_AGENT, "Accept": "*/*"}
        if headers:
            h.update(headers)
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        last_err: Optional[Exception] = None
        for attempt in range(4):
            try:
                return urllib.request.urlopen(req, timeout=timeout)
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last_err = e
                if attempt < 3:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise
        raise last_err if last_err else RuntimeError("HTTP error")

    @staticmethod
    def get_json(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> Any:
        with _Net.request(url, headers=headers, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    @staticmethod
    def get_text(url: str, headers: Optional[Dict[str, str]] = None, timeout: int = 60) -> str:
        with _Net.request(url, headers=headers, timeout=timeout) as r:
            raw = r.read()
            enc = r.headers.get_content_charset() or "utf-8"
        try:
            return raw.decode(enc, errors="replace")
        except LookupError:
            return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
#                            СКАЧИВАНИЕ С ПРОДОЛЖЕНИЕМ
# --------------------------------------------------------------------------- #


def download(
    url: str,
    dest: Path,
    label: str = "",
    headers: Optional[Dict[str, str]] = None,
    expected_sha256: Optional[str] = None,
) -> Path:
    """Скачивает url в dest c прогресс-баром, докачивает .part при обрыве.

    Атомарно переименовывает .part -> dest после успешной загрузки.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    label = label or dest.name
    headers = dict(headers or {})

    total = 0
    try:
        req_head = urllib.request.Request(url, method="HEAD",
                                          headers={"User-Agent": USER_AGENT, **headers})
        with urllib.request.urlopen(req_head, timeout=30) as r:
            total = int(r.headers.get("Content-Length") or 0)
            if r.headers.get("Accept-Ranges", "").lower() != "bytes":
                pass
    except urllib.error.HTTPError as e:
        if e.code in (403, 401):
            raise RuntimeError(
                f"HTTP {e.code}: доступ запрещён. Возможно, модель gated — "
                "примите лицензию на странице модели и задайте токен HF_TOKEN."
            ) from e
    except Exception:
        total = 0

    bar = ProgressBar(label)
    try:
        import hashlib
        sha = hashlib.sha256()
    except Exception:  # экзотика без hashlib — просто без проверки
        sha = None

    have = part.stat().st_size if part.exists() else 0
    if total and have > total:
        part.unlink(missing_ok=True)
        have = 0
    if total and have == total and total > 0:
        bar.finish(f"  {label}: уже скачано ({fmt_bytes(total)})")
    else:
        if have and sha is not None:
            with open(part, "rb") as f:
                while True:
                    chunk = f.read(1024 * 1024)
                    if not chunk:
                        break
                    sha.update(chunk)
        elif have:
            have = 0

        req_headers = {"User-Agent": USER_AGENT, **headers}
        if have:
            req_headers["Range"] = f"bytes={have}-"
        req = urllib.request.Request(url, headers=req_headers)
        try:
            resp = urllib.request.urlopen(req, timeout=60)
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:
                ok(f"  {label}: файл уже полностью скачан")
                have = total or have
                resp = None
            elif e.code in (401, 403):
                raise RuntimeError(
                    f"HTTP {e.code}: доступ запрещён. Для gated-моделей примите лицензию "
                    "на странице модели и задайте токен (--token или env HF_TOKEN)."
                ) from e
            else:
                raise
        if resp is not None:
            with resp:
                status = getattr(resp, "status", 200)
                if status != 206:
                    # сервер не умеет Range — начинаем файл заново
                    have = 0
                    try:
                        import hashlib
                        sha = hashlib.sha256()
                    except Exception:
                        sha = None
                remain = 0
                try:
                    remain = int(resp.headers.get("Content-Length") or 0)
                except ValueError:
                    remain = 0
                if remain:
                    total = have + remain
                mode = "ab" if have else "wb"
                t0 = time.time()
                last_t = t0
                last_n = have
                speed = 0.0
                with open(part, mode) as fh:
                    while True:
                        chunk = resp.read(1024 * 512)
                        if not chunk:
                            break
                        fh.write(chunk)
                        have += len(chunk)
                        if sha is not None:
                            sha.update(chunk)
                        now = time.time()
                        if now - last_t >= 0.4:
                            speed = (have - last_n) / (now - last_t)
                            last_t, last_n = now, have
                        bar.update(have, total, speed)
                bar.update(have, total or have, speed)
        bar.finish(f"  {C.BGREEN}✔{C.RESET} {label}: скачано ({fmt_bytes(have)})")

    if expected_sha256 and sha is not None and total:
        digest = sha.hexdigest()
        if digest.lower() != expected_sha256.lower():
            part.unlink(missing_ok=True)
            raise RuntimeError(
                f"Контрольная сумма не совпала для {label}: {digest[:12]}… != "
                f"{expected_sha256[:12]}… — файл удалён, попробуйте снова."
            )

    os.replace(part, dest)
    if not total:
        total = dest.stat().st_size
    return dest


# --------------------------------------------------------------------------- #
#                      HUGGING FACE: ПАРСИНГ ССЫЛОК И ФАЙЛОВ
# --------------------------------------------------------------------------- #

HF_KINDS = {
    # type -> (keyword, message)
    "model": ("model", "Это НЕ файл модели (.gguf): квантованные файлы лежат в разделе "
                       "\"Files and versions\" (вкладка Files), а не в дереве репозитория."),
    "dataset": ("dataset", "Это датасет, а не модель: веса лежат в repo типа model."),
    "space": ("space", "Это Space, а не модель: веса лежат в repo типа model."),
}
HF_KINDS_KEYS = ("model", "dataset", "datasets", "space", "spaces")


@dataclass
class HfRef:
    kind: str                         # model | dataset | space | direct
    endpoint: str
    repo_id: str = ""
    revision: str = "main"
    file: str = ""                    # путь к файлу в репозитории (для kind != direct)
    url: str = ""                     # итоговый url (для direct)

    def describe(self) -> str:
        if self.kind == "direct":
            return self.url
        return f"{self.endpoint}/{self.repo_id}@{self.revision}/{self.file}"


def hf_headers(token: str = "") -> Dict[str, str]:
    token = token or os.environ.get("HF_TOKEN", "") or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    h = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def _parse_query(qs: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in urllib.parse.parse_qsl(qs):
        out[k] = v
    return out


def parse_hf_ref(raw: str) -> HfRef:
    """Разбирает что угодно из буфера обмена: прямая ссылка, страница HF, hf.co/.."""
    s = (raw or "").strip().strip("'\"")
    if not s:
        raise ValueError("Пустая ссылка.")
    if re.match(r"^[a-zA-Z]:[\\/]", s) or s.startswith(("/", "\\")):
        raise ValueError("Это локальный путь, а не ссылка.")
    short = _parse_short_repo(s)
    if short is not None and short.repo_id.split("/")[0].lower() not in HF_KINDS:
        return short
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", s):
        s = "https://" + s
    u = urllib.parse.urlsplit(s)
    host = (u.hostname or "").lower()
    segs = [urllib.parse.unquote(x) for x in u.path.split("/") if x]
    query = _parse_query(u.query)
    fname = query.get("filename") or ""

    def file_from(filename: str) -> str:
        fn = filename.strip()
        if segs and (fn.startswith("/") or "/" in fn):
            return fn.lstrip("/")
        return fn

    if host in ("huggingface.co", "www.huggingface.co", "hf.co", "www.hf.co", "hf-mirror.com"):
        return _parse_hf_site(segs, query, fname, s)

    if host == "github.com":
        segs_gh = segs
        try:
            i = segs_gh.index("releases")
            if i + 3 < len(segs_gh) and segs_gh[i + 2] == "download":
                asset = "/".join(segs_gh[i + 3:])
                return HfRef("direct", "", "", "main", "",
                             f"https://github.com/{segs_gh[0]}/{segs_gh[1]}/releases/download/"
                             f"{segs_gh[i + 1]}/{urllib.parse.quote(asset)}")
        except ValueError:
            pass
        raise ValueError("Ссылки GitHub пока не поддерживаются напрямую. "
                         "Вставьте прямой URL на файл .gguf.")

    if not u.scheme.startswith("http"):
        raise ValueError(f"Неизвестная схема ссылки: {u.scheme}")
    last = (u.path.rsplit("/", 1)[-1] or "").lower()
    looks_gguf = last.endswith(".gguf") or fname.lower().endswith(".gguf") or ".gguf" in fname
    if looks_gguf and not query.get("page"):
        url = s
        if fname and host == "huggingface.co":
            url = s.split("?")[0]
        return HfRef("direct", "", "", "main", "", url)
    raise ValueError(
        "Не похоже на ссылку на файл модели. Нужна прямая ссылка на .gguf "
        "(или страница модели на huggingface.co)."
    )


def _parse_hf_site(segs: List[str], query: Dict[str, str], fname: str,
                   original: str) -> HfRef:
    """Страницы huggingface.co /*: repo, tree, blob, resolve, ?filename=."""
    ep = HF
    if not segs:
        raise ValueError("В ссылке не указан репозиторий.")
    kind = "model"
    if segs[0] in HF_KINDS_KEYS:
        kind = segs[0][:-1]           # datasets -> dataset, spaces -> space
        segs = segs[1:]
    if len(segs) < 2:
        raise ValueError("В ссылке не указан репозиторий вида owner/name.")
    org, name = segs[0], segs[1]
    if name.endswith(".git"):
        name = name[:-4]
    repo_id = f"{org}/{name}"
    rest = segs[2:]

    # hf.co/owner/repo?filename=path/model.gguf — сразу конкретный файл
    if fname:
        fpath = fname.strip().lstrip("/")
        if fpath.lower().endswith(".gguf"):
            return HfRef("direct", ep, repo_id, "main", fpath,
                         f"{ep}/{repo_id}/resolve/main/{urllib.parse.quote(fpath, safe='/')}")
        raise ValueError(f"В ?filename= указан не .gguf файл: {fpath}")

    if rest and rest[0] in ("resolve", "raw", "legacy", "blob"):
        after = rest[1:]
    elif rest and rest[0] == "tree":
        return HfRef(kind, ep, repo_id, rest[1] if len(rest) > 1 else "main")
    else:
        after = []
    if len(after) < 2:
        return HfRef(kind, ep, repo_id, "main")
    rev, fpath = after[0], "/".join(after[1:])
    return HfRef("direct", ep, repo_id, rev, fpath,
                 f"{ep}/{repo_id}/resolve/{rev}/{urllib.parse.quote(fpath, safe='/')}")


def _parse_short_repo(raw: str) -> Optional[HfRef]:
    """Короткая запись вида Qwen/Qwen2.5-Coder-7B-Instruct-GGUF."""
    m = re.fullmatch(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)/([A-Za-z0-9][A-Za-z0-9._-]*)\s*", raw or "")
    if not m:
        return None
    return HfRef("model", HF, f"{m.group(1)}/{m.group(2)}", "main")


@dataclass
class HfTreeEntry:
    path: str
    size: int

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]

    @property
    def quant(self) -> str:
        n = self.name.upper()
        m = re.search(r"-(IQ\d+_[A-Z0-9]+|Q\d+_K_[A-Z0-9]+|Q\d+_\d+|Q\d+_\d+_[A-Z0-9]+|Q\d+_[A-Z]+|MXFP\d+_\d+|Q\d+K|BF16|F16|F32)(?:[-.]|$)", n)
        return m.group(1) if m else ""


def hf_tree(endpoint: str, repo_id: str, token: str = "") -> List[HfTreeEntry]:
    data = _Net.get_json(f"{endpoint}/api/models/{repo_id}/tree/main",
                         headers=hf_headers(token))
    out: List[HfTreeEntry] = []
    if isinstance(data, list):
        for e in data:
            if isinstance(e, dict) and e.get("type") == "file":
                out.append(HfTreeEntry(e.get("path", ""), int(e.get("size") or 0)))
    return out


def hf_resolve(ep: HfRef, token: str = "") -> HfRef:
    """Из ссылки на репозиторий выбирает .gguf (с уточнением у пользователя)."""
    if ep.kind != "model":
        kw, msg = HF_KINDS.get(ep.kind, ("model", "Неподдерживаемый тип репозитория."))
        raise RuntimeError(msg)
    if ep.file:
        return HfRef("direct", ep.endpoint, ep.repo_id, ep.revision, ep.file,
                     f"{ep.endpoint}/{ep.repo_id}/resolve/{ep.revision}/{urllib.parse.quote(ep.file)}")
    note(f"Смотрю файлы репозитория {C.BOLD}{ep.repo_id}{C.RESET}…")
    entries = [e for e in hf_tree(ep.endpoint, ep.repo_id, token) if e.path.lower().endswith(".gguf")]
    gguf = [e for e in entries if not e.name.startswith(".")]
    split = [e for e in gguf if re.search(r"-\d{5}-of-\d{5}\.gguf$", e.name)]
    unsplit = [e for e in gguf if e not in split]
    if not unsplit:
        if split:
            raise RuntimeError(
                "В репозитории только разбитые на части файлы (-00001-of-000NN.gguf). "
                "llama-server требует одиночный GGUF — выберите квант со слитым файлом."
            )
        raise RuntimeError(
            f"В {ep.repo_id} нет файлов .gguf. Это обычный формат transformers "
            "(safetensors). Нужна GGUF-версия модели — поищите репозиторий с суффиксом "
            "-GGUF у того же автора или через hf.co/models?search=<имя>+GGUF."
        )
    default = 0
    for i, e in enumerate(unsplit):
        if re.search(r"Q4_K_M", e.name, re.I):
            default = i
    print()
    print(f"  {C.BOLD}Найденные GGUF-файлы:{C.RESET}")
    rows = [(str(i + 1), e.path, f"{fmt_bytes(e.size)}  {e.quant}") for i, e in enumerate(unsplit)]
    fmt_table(rows, ("№", "файл", "размер / квант"))
    idx = choose("Какой файл скачать", [r[1] for r in rows], default + 1)
    e = unsplit[idx]
    return HfRef("direct", ep.endpoint, ep.repo_id, ep.revision, e.path,
                 f"{ep.endpoint}/{ep.repo_id}/resolve/{ep.revision}/{urllib.parse.quote(e.path)}")


def model_filename_for(ref: HfRef) -> str:
    if ref.file:
        return re.sub(r"[^A-Za-z0-9._-]+", "_", ref.file.rsplit("/", 1)[-1])
    name = urllib.parse.unquote(urllib.parse.urlsplit(ref.url).path.rsplit("/", 1)[-1]) or "model.gguf"
    if ref.repo_id and not name.startswith(ref.repo_id.split("/")[-1]):
        name = ref.repo_id.replace("/", "_") + "_" + name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


# --------------------------------------------------------------------------- #
#                               LLAMA.CPP: УСТАНОВКА
# --------------------------------------------------------------------------- #

_PRETTY_OS = {"Windows": "Windows", "Linux": "Linux", "Darwin": "macOS"}


@dataclass
class LlamaInstall:
    server: Path
    mode: str          # "release" | "path"
    info: str = ""


def find_llamacpp(P: Dict[str, Path]) -> Optional[LlamaInstall]:
    """Ищет llama-server: сначала в нашей папке (rc-сборки), затем в PATH."""
    exe = "llama-server.exe" if IS_WINDOWS else "llama-server"
    bin_dir = P["bin"]
    if bin_dir.is_dir():
        candidates = list(bin_dir.rglob(exe))
        if candidates:
            return LlamaInstall(sorted(candidates, key=lambda p: len(p.parts))[0], "release")
    found = shutil.which("llama-server")
    if found:
        return LlamaInstall(Path(found), "path", "найден в PATH")
    return None


def _libcpu_variant() -> str:
    """Нужен ли fallback на CPU-сборку по базовой архитектуре процессора."""
    if IS_WINDOWS:
        # modern prebuilt для Windows требуют AVX2 (Haswell 2013+)
        return "cpu"
    if IS_LINUX:
        return "cpu"
    if IS_MAC:
        return ""
    return ""


def _platform_candidates() -> List[Tuple[str, str]]:
    """Список (label, asset_name) по приоритету: сначала быстрый вариант, потом CPU."""
    if IS_WINDOWS:
        base = ["win-cpu-x64" if platform.machine().lower() not in ("arm64", "aarch64")
                else "win-cpu-arm64"]
        if platform.machine().lower() in ("arm64", "aarch64"):
            return ["win-cpu-arm64"]
        return ["win-vulkan-x64", "win-cpu-x64"]
    if IS_MAC:
        return ["macos-arm64" if platform.machine() == "arm64" else "macos-x64"]
    if IS_LINUX:
        m = platform.machine().lower()
        arch = "x64" if m in ("x86_64", "amd64") else ("arm64" if m in ("aarch64", "arm64") else "")
        if not arch:
            return []
        return [f"ubuntu-vulkan-{arch}", f"ubuntu-{arch}"]
    return []


def _asset_name_ok(name: str, cand: str) -> bool:
    n = name.lower()
    base = cand.replace("_", "-")
    if not n.startswith("llama-") or base not in n:
        return False
    # исключаем варианты для других архитектур/бэкендов
    if "-sycl" in n and "-sycl" not in base:
        return False
    if "-cuda" in n or "+cuda" in n:
        return False
    return n.endswith(".zip") or n.endswith(".tar.gz")


def _release_assets(api_json: dict) -> List[dict]:
    assets = api_json.get("assets") or []
    return [a for a in assets if isinstance(a, dict) and a.get("browser_download_url")]


def _releases_list(limit: int = 30) -> List[dict]:
    """Список релизов llama.cpp: сначала стабильные (с бинарниками), потом nightly."""
    try:
        data = _Net.get_json(f"https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page={limit}",
                             headers={"Accept": "application/vnd.github+json"})
        if isinstance(data, list):
            return [r for r in data if isinstance(r, dict)]
    except Exception as e:
        warn(f"Список релизов GitHub недоступен ({e}), беру последний.")
    try:
        return [_Net.get_json(LLAMA_RELEASES_API, headers={"Accept": "application/vnd.github+json"})]
    except Exception:
        return []


def install_llamacpp_auto(P: Dict[str, Path], token: str = "") -> LlamaInstall:
    """Скачивает официальную сборку llama.cpp c GitHub Releases и распаковывает её.

    Берём свежий релиз, в котором есть бинарники для нашей платформы:
    стабильные релизы в приоритете, затем nightly (bXXXXX).
    """
    rule("Установка llama.cpp")
    osname = platform.system()
    cands = _platform_candidates()
    if not cands:
        raise RuntimeError(
            f"Для {osname}/{platform.machine()} нет готовой сборки llama.cpp. "
            f"Соберите вручную по инструкции: {LLAMA_BUILD_PAGE}"
        )
    note("Спрашиваю у GitHub свежие релизы llama.cpp…")
    releases = _releases_list(30)
    if not releases:
        warn("GitHub API недоступен — пробую прямые ссылки latest/download.")

    def pick(release: dict) -> Optional[Tuple[str, str, str]]:
        tag = str(release.get("tag_name") or "")
        assets = _release_assets(release)
        for cand in cands:
            for a in assets:
                if _asset_name_ok(a["name"], cand):
                    return tag, a["name"], a["browser_download_url"]
        if not assets and tag:
            for cand in cands:
                name = (f"llama-{tag}-bin-{cand}.zip" if IS_WINDOWS
                        else f"llama-{tag}-bin-{cand}.tar.gz")
                url = f"{LLAMA_RELEASES_PAGE}/{name}"
                try:
                    with _Net.request(url, method="HEAD", timeout=20) as r:
                        if r.status == 200:
                            return tag, name, url
                except Exception:
                    continue
        return None

    stable = [r for r in releases if not r.get("prerelease") and not r.get("draft")]
    nightly = [r for r in releases if r.get("prerelease") and not r.get("draft")]
    chosen: Optional[Tuple[str, str, str]] = None
    for rel in stable + nightly:
        chosen = pick(rel)
        if chosen:
            break
    if not chosen:
        raise RuntimeError(
            "Не нашёл подходящую сборку llama.cpp (" + ", ".join(cands) + "). "
            f"Скачайте вручную: {LLAMA_RELEASES_ALL} и положите llama-server в {P['bin']}"
        )
    tag, asset_name, asset_url = chosen
    if IS_WINDOWS and asset_name.endswith(".tar.gz"):
        raise RuntimeError(f"Неожиданный формат ассета: {asset_name}")
    ok(f"Сборка: {C.BOLD}{asset_name}{C.RESET} (релиз {tag or '?'})")

    archive = P["tmp"] / asset_name
    note(f"Скачиваю → {archive.parent}")
    download(asset_url, archive, label=asset_name[-26:], headers=hf_headers(token))

    bin_dir = P["bin"]
    if bin_dir.exists():
        shutil.rmtree(bin_dir, ignore_errors=True)
    bin_dir.mkdir(parents=True, exist_ok=True)
    note("Распаковываю…")
    try:
        if asset_name.endswith(".zip"):
            with zipfile.ZipFile(archive) as z:
                z.extractall(bin_dir)
        else:
            with tarfile.open(archive, "r:gz") as t:
                # фильтр «data» закрывает path traversal и опасные спецфайлы
                if hasattr(tarfile, "data_filter"):
                    t.extractall(bin_dir, filter="data")
                else:
                    t.extractall(bin_dir)
    finally:
        archive.unlink(missing_ok=True)

    _fix_permissions(bin_dir)
    inst = find_llamacpp(P)
    if not inst:
        raise RuntimeError(
            f"Скачал и распаковал, но llama-server не найден в {bin_dir}. "
            f"Проверьте содержимое папки или скачайте сборку вручную: {LLAMA_RELEASES_ALL}"
        )
    try:
        ver = subprocess.run([str(inst.server), "--version"], capture_output=True,
                             text=True, timeout=20)
        lines = (ver.stdout or ver.stderr or "").strip().splitlines()
        ok(f"llama.cpp установлен: {inst.server}")
        if lines:
            print(f"  {C.GREY}{lines[0]}{C.RESET}")
    except Exception:
        ok(f"llama.cpp установлен: {inst.server}")
    return inst


def _fix_permissions(bin_dir: Path) -> None:
    if IS_WINDOWS:
        return
    for p in bin_dir.rglob("*"):
        if p.is_file():
            try:
                dylib = p.suffix in (".dylib", ".so") or ".so." in p.name
                if dylib or p.name.startswith(("llama-", "ggml-")):
                    p.chmod(p.stat().st_mode | 0o755)
            except OSError:
                pass


def manual_install(P: Dict[str, Path]) -> LlamaInstall:
    note("Открываю страницу релизов llama.cpp в браузере…")
    try:
        _open_browser(LLAMA_RELEASES_ALL)
    except Exception:
        pass
    print(f"  Скачайте архив {C.BOLD}bin-<платформа>{C.RESET} под вашу ОС")
    print(f"  (например для Windows x64: win-vulkan-x64.zip или win-cpu-x64.zip)")
    print(f"  и вставьте путь к архиву (.zip / .tar.gz) или к самому llama-server.")
    while True:
        p = ask("Путь к архиву/исполняемому файлу")
        path = Path(p).expanduser()
        try:
            if not path.exists():
                err(f"Не найдено: {path}")
                continue
            if path.is_dir():
                src = next((q for q in path.rglob("llama-server*")), None)
                if not src:
                    err("В папке нет llama-server.")
                    continue
                target = P["bin"] / src.parent.name
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(src.parent, target)
            elif path.suffix in (".zip", ".tar.gz", ".tgz") or path.name.endswith(".tar.gz"):
                dest = P["bin"]
                shutil.rmtree(dest, ignore_errors=True)
                dest.mkdir(parents=True, exist_ok=True)
                if path.name.endswith(".zip"):
                    with zipfile.ZipFile(path) as z:
                        z.extractall(dest)
                else:
                    with tarfile.open(path, "r:gz") as t:
                        if hasattr(tarfile, "data_filter"):
                            t.extractall(dest, filter="data")
                        else:
                            t.extractall(dest)
                _fix_permissions(dest)
            else:
                target_dir = P["bin"] / "manual"
                target_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target_dir / path.name)
                _fix_permissions(target_dir)
        except Exception as e:
            err(f"Не удалось: {e}")
            continue
        inst = find_llamacpp(P)
        if inst:
            ok(f"llama.cpp готов: {inst.server}")
            return inst
        err("llama-server всё ещё не найден. Попробуйте другой путь.")


def ensure_llamacpp(P: Dict[str, Path], cfg: Dict[str, Any], auto_yes: bool = False) -> LlamaInstall:
    inst = find_llamacpp(P)
    if inst:
        ok(f"llama.cpp найден: {inst.server}" + (f" ({inst.info})" if inst.info else ""))
        return inst

    rule("llama.cpp не найден")
    print("  Без llama-server локальная модель не запустится.")
    print("  Можно скачать официальную сборку автоматически (≈150–300 МБ)…")
    print(f"  {C.GREY}Версия: последний релиз с {LLAMA_RELEASES_ALL}{C.RESET}")
    print(f"  {C.GREY}Сборка будет установлена в: {P['bin']}{C.RESET}")
    if auto_yes or ask_yes_no("Скачать llama.cpp автоматически?", True):
        try:
            return install_llamacpp_auto(P, cfg.get("hf_token", ""))
        except Exception as e:
            err(f"Автоустановка не удалась: {e}")
            warn("Переходим к ручной установке.")
            return manual_install(P)
    return manual_install(P)


def _open_browser(url: str) -> bool:
    import webbrowser
    return webbrowser.open(url)


# --------------------------------------------------------------------------- #
#                         LLAMA-SERVER: ПРОЦЕСС И API
# --------------------------------------------------------------------------- #


def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ServerProcess:
    def __init__(self, inst: LlamaInstall, model: Path, cfg: Dict[str, Any]) -> None:
        self.inst = inst
        self.model = model
        self.cfg = cfg
        self.port = int(cfg.get("port") or 0) or free_port()
        self.base = f"http://127.0.0.1:{self.port}"
        self.proc: Optional[subprocess.Popen] = None
        self.log_path = home_paths(cfg["_home"])["home"] / "llama-server.log"
        self._logfh = None

    def _args(self) -> List[str]:
        c = self.cfg
        args = [
            str(self.inst.server),
            "-m", str(self.model),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-c", str(int(c.get("ctx") or 0)),
            "-ngl", str(int(c.get("gpu_layers", -1))),
            "--jinja",
        ]
        threads = int(c.get("threads") or 0)
        if threads > 0:
            args += ["-t", str(threads)]
        fa = str(c.get("flash_attn") or "auto")
        if fa == "on":
            args += ["-fa", "on"]
        elif fa == "off":
            args += ["-fa", "off"]
        if c.get("no_mmap"):
            args += ["--no-mmap"]
        if c.get("api_key"):
            args += ["--api-key", str(c["api_key"])]
        extra = c.get("extra_args") or []
        if isinstance(extra, list):
            args += [str(a) for a in extra]
        return args

    def start(self) -> None:
        args = self._args()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logfh = open(self.log_path, "w", encoding="utf-8", errors="replace")
        note(f"Запуск: {C.GREY}{' '.join(args)}{C.RESET}")
        creationflags = 0
        if IS_WINDOWS:
            creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            # лог llama-server всегда пишется в файл (stdout+stderr),
            # иначе при падении пользователь не поймёт причину
            self.proc = subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=self._logfh,
                stderr=subprocess.STDOUT,
                creationflags=creationflags,
            )
        except OSError as e:
            raise RuntimeError(f"Не удалось запустить {args[0]}: {e}")

    def _models_ok(self) -> bool:
        try:
            data = _Net.get_json(self.base + "/v1/models", headers=self._auth(), timeout=3)
            return bool(data.get("data"))
        except Exception:
            return False

    def _auth(self) -> Dict[str, str]:
        key = self.cfg.get("api_key")
        return {"Authorization": f"Bearer {key}"} if key else {}

    def wait_ready(self, timeout: float = 900) -> bool:
        t0 = time.time()
        bar = ProgressBar("загрузка модели")
        if not _live_terminal():
            note("Загрузка модели в память…")
        while time.time() - t0 < timeout:
            if self.proc is not None and self.proc.poll() is not None:
                bar.finish()
                tail = self.log_tail(40)
                err(f"llama-server завершился с кодом {self.proc.returncode}.")
                if tail:
                    print(C.GREY + tail + C.RESET)
                return False
            if self._models_ok():
                bar.finish()
                return True
            if _live_terminal():
                sp = SPINNER_FRAMES[int(time.time() * 8) % len(SPINNER_FRAMES)]
                sys.stdout.write(f"\r  {C.BCYAN}{sp}{C.RESET} загрузка модели в память… "
                                 f"{int(time.time() - t0)} с ")
                sys.stdout.flush()
            elif not _live_terminal() and (int(time.time() - t0) + 1) % 10 == 0 and time.time() - t0 > 1:
                print(f"    … жду модель: {int(time.time() - t0)} с", flush=True)
            time.sleep(0.5)
        bar.finish()
        warn("Модель не стала доступна за отведённое время (смотрите лог).")
        return self._models_ok()

    def log_tail(self, n: int = 30) -> str:
        try:
            lines = self.log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n:])
        except Exception:
            return ""

    def stop(self) -> None:
        p = self.proc
        if not p or p.poll() is not None:
            return
        try:
            p.terminate()
            try:
                p.wait(8)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(5)
        except Exception:
            pass
        finally:
            if self._logfh:
                try:
                    self._logfh.close()
                except Exception:
                    pass


class ChatClient:
    """Небольшой клиент OpenAI-совместимого /v1/chat/completions llama.cpp."""

    def __init__(self, base: str, api_key: str = "", temperature: float = 0.2,
                 timeout: float = 600) -> None:
        self.base = base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.timeout = timeout
        self._lock = threading.Lock()
        self._cancel = threading.Event()

    def cancel(self) -> None:
        self._cancel.set()

    def stream(self, messages: List[Dict[str, Any]],
               on_token: Callable[[str], None]) -> Tuple[str, Dict[str, Any], List[dict]]:
        payload = {
            "model": "local",
            "messages": messages,
            "stream": True,
            "temperature": self.temperature,
            "cache_prompt": True,
            "return_progress": True,
            "stream_options": {"include_usage": True},
        }
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.base + "/v1/chat/completions",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers=headers, method="POST")
        self._cancel.clear()
        parts: List[str] = []
        usage: Dict[str, Any] = {}
        timings: Dict[str, Any] = {}
        first_ts: Optional[float] = None
        try:
            resp = urllib.request.urlopen(req, timeout=self.timeout)
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(_explain_http_error(e.code, body))
        except urllib.error.URLError as e:
            raise RuntimeError(f"Не удалось связаться с llama-server: {e.reason}")

        with resp:
            for raw in resp:
                if self._cancel.is_set():
                    break
                line = raw.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if not isinstance(ev, dict):
                    continue
                if ev.get("timings"):
                    timings = ev["timings"] or {}
                if ev.get("usage"):
                    usage = ev["usage"] or {}
                if ev.get("error"):
                    raise RuntimeError(str(ev["error"]))
                for ch in ev.get("choices") or []:
                    if not isinstance(ch, dict):
                        continue
                    delta = ch.get("delta") or {}
                    content = delta.get("content") or (ch.get("message") or {}).get("content")
                    if not content:
                        continue
                    if first_ts is None:
                        first_ts = time.time()
                    parts.append(content)
                    on_token(content)
        text = "".join(parts)
        extra = {
            "first_token_ts": first_ts,
            "usage": usage,
            "timings": timings,
            "prompt_tokens": usage.get("prompt_tokens") or timings.get("prompt_n"),
            "completion_tokens": usage.get("completion_tokens") or timings.get("predicted_n"),
            "tok_per_s": timings.get("predicted_per_second"),
        }
        return text, extra, payload["messages"]


def _explain_http_error(code: int, body: str) -> str:
    snippet = strip_ansi(body).strip()
    if len(snippet) > 600:
        snippet = snippet[:600] + "…"
    if code == 400:
        return (f"llama-server ответил 400 (неверный запрос). Чаще всего это значит, что "
                f"контекст модели переполнен.\n{_wrap(snippet)}")
    if code == 404:
        return f"llama-server ответил 404 — неверный адрес API.\n{_wrap(snippet)}"
    return f"llama-server вернул HTTP {code}.\n{_wrap(snippet)}"


def _wrap(text: str, width: int = 100) -> str:
    out = []
    for ln in text.splitlines()[:20]:
        while len(ln) > width:
            out.append(ln[:width])
            ln = ln[width:]
        out.append(ln)
    return "\n".join(out)


# --------------------------------------------------------------------------- #
#                             АГЕНТ: ЖУРНАЛ И ИНСТРУМЕНТЫ
# --------------------------------------------------------------------------- #


@dataclass
class Tool:
    name: str
    description: str
    parameters: Dict[str, Any]
    run: Callable[..., Tuple[bool, str]]
    write: bool = False


@dataclass
class JournalEntry:
    files: List[Dict[str, Any]] = field(default_factory=list)


class UndoJournal:
    """Хранит состояние файлов до изменения, чтобы откатывать правки агента."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.stash = self.root / "stash"
        self.stash.mkdir(exist_ok=True)
        self.log = self.root / "journal.jsonl"
        self._counter = 0
        self.group = "0"
        self._pending: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def _stash(self, path: Path) -> Optional[str]:
        if not path.exists() or path.is_dir():
            return None
        self._counter += 1
        name = f"{int(time.time() * 1000)}_{self._counter}_{re.sub(r'[^A-Za-z0-9._-]+', '_', path.name)[:60]}"
        dest = self.stash / name
        try:
            shutil.copy2(path, dest)
            return str(dest)
        except Exception:
            return None

    def record(self, tool: str, path: Path, action: str) -> None:
        """Запоминает состояние файла ДО изменения; пишется на диск в flush()."""
        try:
            with self._lock:
                group = self.group or "0"
                self._counter += 1
                entry = {
                    "ts": datetime.now().isoformat(timespec="seconds"),
                    "group": group,
                    "seq": self._counter,
                    "tool": tool,
                    "path": str(path),
                    "action": action,
                    "stash": self._stash(path),
                    "existed": path.exists(),
                }
                self._pending.append(entry)
        except Exception as e:
            warn(f"Журнал правок: {e}")

    def flush(self) -> None:
        """Сохраняет накопленные записи; все правки одного хода = одна группа отката."""
        with self._lock:
            pending, self._pending = self._pending, []
        if not pending:
            return
        try:
            # одна группа на весь ход агента
            for e in pending:
                e["group"] = "turn"
            with open(self.log, "a", encoding="utf-8") as f:
                for e in pending:
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        except Exception as e:
            warn(f"Журнал правок: {e}")

    def _load(self) -> List[Dict[str, Any]]:
        if not self.log.exists():
            return []
        out: List[Dict[str, Any]] = []
        for ln in self.log.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(ln)
                if isinstance(e, dict):
                    out.append(e)
            except json.JSONDecodeError:
                continue
        return out

    def _save(self, entries: List[Dict[str, Any]]) -> None:
        tmp = self.log.with_suffix(".jsonl.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for e in entries[-2000:]:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, self.log)

    def undo_last(self) -> int:
        """Откатывает всё, что агент сделал в последнем ходе (одна группа)."""
        entries = self._load()
        if not entries:
            return 0
        last_group = None
        for idx in range(len(entries) - 1, -1, -1):
            if not entries[idx].get("undone"):
                last_group = entries[idx].get("group")
                break
        if last_group is None:
            return 0
        group_set = [e for e in entries
                     if not e.get("undone") and e.get("group") == last_group]
        restored = 0
        for e in reversed(group_set):
            p = Path(e["path"])
            stash = e.get("stash")
            try:
                if e.get("existed") and stash and Path(stash).exists():
                    p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(stash, p)
                    restored += 1
                elif not e.get("existed") and p.exists() and p.is_file():
                    p.unlink()
                    restored += 1
            except Exception as ex:
                warn(f"Не удалось откатить {p}: {ex}")
        ids = {id(e) for e in group_set}
        for e in entries:
            if id(e) in ids:
                e["undone"] = True
        self._save(entries)
        # чистим стэш этой группы
        try:
            for e in group_set:
                if e.get("stash"):
                    Path(e["stash"]).unlink(missing_ok=True)
        except Exception:
            pass
        return restored


class AgentTools:
    def __init__(self, workdir: Path, journal: UndoJournal, approve: bool = True,
                 on_approve: Optional[Callable[[str, str], bool]] = None,
                 on_event: Optional[Callable[[str], None]] = None) -> None:
        self.workdir = workdir.resolve()
        self.journal = journal
        self.approve = approve
        self.on_approve = on_approve or (lambda desc, body: True)
        self.on_event = on_event or (lambda msg: None)
        self.todos: List[Dict[str, str]] = []
        self.group = ""
        self.readonly = False
        self.tools: Dict[str, Tool] = {}
        self._register()

    # ----- инфраструктура -------------------------------------------------- #
    def new_turn(self) -> None:
        self.group = datetime.now().strftime("%H%M%S") + f"_{random.randint(0, 999):03d}"

    def _resolve(self, p: str) -> Path:
        if not p or not str(p).strip():
            raise ValueError("Не указан путь.")
        raw = str(p).strip().strip('"').strip("'")
        # если модель прислала абсолютный путь внутри другой копии проекта — оставляем
        cand = Path(os.path.expanduser(raw))
        if not cand.is_absolute():
            cand = self.workdir / cand
        return cand.resolve()

    def _rel(self, p: Path) -> str:
        try:
            return str(p.relative_to(self.workdir)).replace("\\", "/") or "."
        except ValueError:
            return str(p)

    def _gate(self, desc: str, body: str, write: bool) -> bool:
        if not (self.approve and write) or self.readonly:
            return True
        return self.on_approve(desc, body)

    def _write_policy(self) -> str:
        return ("спрашивать" if self.approve else "не спрашивать")

    def describe(self) -> List[Dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in self.tools.values()]

    def _register(self) -> None:
        def add(t: Tool) -> None:
            self.tools[t.name] = t

        add(Tool("read_file", "Прочитать текстовый файл (если строк больше лимита — укажи offset/limit).",
                 {"type": "object", "properties": {
                     "path": {"type": "string", "description": "Путь к файлу"},
                     "offset": {"type": "integer", "description": "С какой строки начать (1)"},
                     "limit": {"type": "integer", "description": "Сколько строк прочитать"},
                 }, "required": ["path"]}, self.t_read))
        add(Tool("write_file", "Создать файл или полностью перезаписать его.",
                 {"type": "object", "properties": {
                     "path": {"type": "string"},
                     "content": {"type": "string", "description": "Полное новое содержимое"},
                 }, "required": ["path", "content"]}, self.t_write, write=True))
        add(Tool("edit_file", "Заменить точный фрагмент текста в файле (old_string → new_string). "
                              "old_string должен встречаться ровно один раз.",
                 {"type": "object", "properties": {
                     "path": {"type": "string"},
                     "old_string": {"type": "string"},
                     "new_string": {"type": "string"},
                     "replace_all": {"type": "boolean"},
                 }, "required": ["path", "old_string", "new_string"]}, self.t_edit, write=True))
        add(Tool("apply_patch", "Применить unified diff (можно несколько файлов за раз). "
                                "Пути без префикса a/ b/ считаются относительными.",
                 {"type": "object", "properties": {
                     "patch": {"type": "string", "description": "Текст ---/+++/@@ патча"},
                 }, "required": ["patch"]}, self.t_patch, write=True))
        add(Tool("list_dir", "Показать содержимое папки (с размерами).",
                 {"type": "object", "properties": {
                     "path": {"type": "string", "description": "По умолчанию корень проекта"},
                 }}, self.t_list))
        add(Tool("glob", "Найти файлы по шаблону, например '**/*.py'.",
                 {"type": "object", "properties": {
                     "pattern": {"type": "string"},
                     "path": {"type": "string"},
                 }, "required": ["pattern"]}, self.t_glob))
        add(Tool("grep", "Поиск регулярного выражения по содержимому файлов.",
                 {"type": "object", "properties": {
                     "pattern": {"type": "string"},
                     "path": {"type": "string"},
                     "include": {"type": "string", "description": "Маска файлов, например *.py"},
                 }, "required": ["pattern"]}, self.t_grep))
        add(Tool("run_shell", "Выполнить команду оболочки в рабочей папке и вернуть вывод.",
                 {"type": "object", "properties": {
                     "command": {"type": "string"},
                     "timeout": {"type": "integer", "description": "Секунды (по умолчанию 120)"},
                 }, "required": ["command"]}, self.t_shell, write=True))
        add(Tool("web_fetch", "Скачать страницу/файл по http(s) и вернуть текст (HTML → текст).",
                 {"type": "object", "properties": {
                     "url": {"type": "string"},
                 }, "required": ["url"]}, self.t_web))
        add(Tool("todo_write", "Записать список задач и статусов (pending/in_progress/completed).",
                 {"type": "object", "properties": {
                     "todos": {"type": "array", "items": {"type": "object"}},
                 }, "required": ["todos"]}, self.t_todo))

    # ----- реализация ------------------------------------------------------ #
    def t_read(self, path: str = "", offset: int = 0, limit: int = 0, **_: Any) -> Tuple[bool, str]:
        p = self._resolve(path)
        if not p.is_file():
            if p.is_dir():
                return False, f"{self._rel(p)} — это папка, используй list_dir."
            return False, f"Файл не найден: {self._rel(p)}"
        raw = p.read_bytes()
        if b"\x00" in raw[:8000]:
            return False, f"{self._rel(p)} — бинарный файл ({fmt_bytes(len(raw))})."
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        start = max(0, (int(offset) - 1) if offset else 0)
        end = start + int(limit) if limit else len(lines)
        chunk = lines[start:end]
        out = "\n".join(f"{start + i + 1:6d}| {ln}" for i, ln in enumerate(chunk))
        truncated = False
        if len(out) > MAX_READ_CHARS:
            out = out[:MAX_READ_CHARS]
            truncated = True
        footer = f"\n\n[{self._rel(p)}: строки {start + 1}-{start + len(chunk)} из {len(lines)}"
        if truncated or end < len(lines):
            footer += f"; прочитай дальше с offset={start + len(chunk) + 1}"
        footer += "]"
        return True, out + footer

    def t_write(self, path: str = "", content: str = "", **_: Any) -> Tuple[bool, str]:
        p = self._resolve(path)
        existed = p.exists()
        preview = content if len(content) < 1500 else content[:1500] + "\n… (обрезано)"
        if not self._gate(f"{'перезаписать' if existed else 'создать'} {self._rel(p)}",
                          preview, True):
            return False, "Пользователь отклонил запись файла."
        p.parent.mkdir(parents=True, exist_ok=True)
        self.journal.record("write_file", p, "edit" if existed else "create")
        old_nl = "\r\n" in p.read_text(encoding="utf-8", errors="replace") if existed and p.stat().st_size < 5_000_000 else IS_WINDOWS
        data = content.replace("\r\n", "\n")
        if old_nl:
            data = data.replace("\n", "\r\n")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(data)
        nlines = data.count("\n") + 1
        return True, f"{'Перезаписан' if existed else 'Создан'} {self._rel(p)} ({nlines} строк, {fmt_bytes(len(data.encode()))})."

    def t_edit(self, path: str = "", old_string: str = "", new_string: str = "",
               replace_all: bool = False, **_: Any) -> Tuple[bool, str]:
        p = self._resolve(path)
        if not p.is_file():
            return False, f"Файл не найден: {self._rel(p)}"
        text = p.read_text(encoding="utf-8", errors="replace")
        if old_string == new_string:
            return False, "old_string и new_string совпадают."
        cnt = text.count(old_string)
        if cnt == 0:
            hint = ""
            if old_string.strip() and text.count(old_string.strip()):
                hint = " (без отступов фрагмент есть — проверь пробелы/отступы)"
            elif old_string.replace("\r\n", "\n") in text.replace("\r\n", "\n"):
                hint = " (нашёл без учёта CRLF — попробуй применить патч через apply_patch)"
            return False, f"Фрагмент не найден в {self._rel(p)}.{hint}"
        if cnt > 1 and not replace_all:
            return False, (f"Фрагмент встречается {cnt} раз. Сделай old_string уникальнее "
                           "или поставь replace_all=true.")
        if not self._gate(f"изменить {self._rel(p)}",
                          f"- {old_string[:800]}\n+ {new_string[:800]}", True):
            return False, "Пользователь отклонил изменение."
        self.journal.record("edit_file", p, "edit")
        crlf = "\r\n" in text
        new_text = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        if crlf:
            # файл был в CRLF: вернуть переводы строк, иначе они смешаются с LF
            new_text = new_text.replace("\r\n", "\n").replace("\n", "\r\n")
        with open(p, "w", encoding="utf-8", newline="") as f:
            f.write(new_text)
        return True, f"Изменён {self._rel(p)} (замен: {cnt if replace_all else 1})."

    def t_patch(self, patch: str = "", **_: Any) -> Tuple[bool, str]:
        hunks = parse_patch(patch)
        if not hunks:
            return False, "Не удалось разобрать патч. Нужны строки --- a/file, +++ b/file и @@."
        results: List[str] = []
        for h in hunks:
            fpath = h.path
            p = self._resolve(fpath)
            if not self._gate(f"применить патч к {self._rel(p)}", patch[:2000], True):
                results.append(f"{self._rel(p)}: отклонено пользователем")
                continue
            if h.is_new:
                old = ""
            else:
                if not p.is_file():
                    results.append(f"{self._rel(p)}: файл не найден")
                    continue
                old = p.read_text(encoding="utf-8", errors="replace")
                if old == "" and h.old_lines and not h.new_lines:
                    results.append(f"{self._rel(p)}: пустой файл")
                    continue
            new, msg = _apply_hunks(old, h.hunks, h.is_new)
            if new is None:
                results.append(f"{self._rel(p)}: {msg}")
                continue
            existed = p.exists()
            self.journal.record("apply_patch", p, "edit" if existed else "create")
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="") as f:
                f.write(new)
            results.append(f"{self._rel(p)}: применено ({len(h.hunks)} ханков)")
        success = all("применено" in r for r in results)
        return success, "\n".join(results)

    def t_list(self, path: str = "", **_: Any) -> Tuple[bool, str]:
        p = self._resolve(path or ".")
        if not p.is_dir():
            return False, f"Не папка: {self._rel(p)}"
        items = []
        try:
            for e in sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
                if e.name in (".git", "node_modules", "__pycache__", ".venv", "venv", ".idea"):
                    items.append(f"  {e.name}/  (скрыто)")
                    continue
                if e.is_dir():
                    try:
                        n = len(list(e.iterdir()))
                    except Exception:
                        n = -1
                    items.append(f"  {e.name}/  ({n} файлов)" if n >= 0 else f"  {e.name}/")
                else:
                    items.append(f"  {e.name}  ({fmt_bytes(e.stat().st_size)})")
                if len(items) > 400:
                    items.append("  … (список обрезан)")
                    break
        except PermissionError as e:
            return False, f"Нет доступа: {e}"
        head = f"{self._rel(p)}:\n" if items else f"{self._rel(p)}: пусто"
        return True, head + "\n".join(items)

    def _walk_files(self, base: Path) -> List[Path]:
        skip = {".git", "node_modules", "__pycache__", ".venv", "venv", ".idea", ".mypy_cache",
                ".pytest_cache", "dist", "build", ".next", "target", ".gradle"}
        files: List[Path] = []
        for root, dirs, fnames in os.walk(base):
            dirs[:] = [d for d in dirs if d not in skip]
            for fn in fnames:
                files.append(Path(root) / fn)
                if len(files) > 20000:
                    return files
        return files

    def t_glob(self, pattern: str = "**/*", path: str = "", **_: Any) -> Tuple[bool, str]:
        base = self._resolve(path or ".")
        if not base.is_dir():
            return False, f"Не папка: {self._rel(base)}"
        matches = sorted(base.glob(pattern), key=lambda p: str(p))[:500]
        files = [f"  {self._rel(m)}" for m in matches if m.is_file()]
        if not files:
            return True, f"По шаблону {pattern} ничего не найдено."
        return True, "\n".join(files) + (f"\n… всего найдено {len(matches)}" if len(matches) >= 500 else "")

    def t_grep(self, pattern: str = "", path: str = "", include: str = "", **_: Any) -> Tuple[bool, str]:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            return False, f"Некорректное регулярное выражение: {e}"
        base = self._resolve(path or ".")
        files = [base] if base.is_file() else self._walk_files(base)
        out: List[str] = []
        scanned = 0
        for f in files:
            if include and not _match_mask(f, include, base):
                continue
            scanned += 1
            if scanned > 5000:
                out.append("… (слишком много файлов, обрезано)")
                break
            try:
                if f.stat().st_size > 4_000_000:
                    continue
                with open(f, "r", encoding="utf-8", errors="strict") as fh:
                    for i, line in enumerate(fh, 1):
                        if rx.search(line):
                            out.append(f"{self._rel(f)}:{i}: {_clip(line.rstrip()[:300], 300)}")
                            if len(out) >= 400:
                                out.append("… (обрезка после 400 совпадений)")
                                return True, "\n".join(out)
            except (UnicodeDecodeError, OSError):
                continue
        return True, ("\n".join(out) if out else "Совпадений не найдено.")

    def t_shell(self, command: str = "", timeout: int = 0, **_: Any) -> Tuple[bool, str]:
        if not command.strip():
            return False, "Пустая команда."
        if not self._gate("выполнить команду", command, True):
            return False, "Пользователь отклонил выполнение команды."
        to = int(timeout) if timeout else SHELL_TIMEOUT
        to = max(5, min(to, 1800))
        self.on_event(f"shell: {command}")
        try:
            res = run_shell_command(command, self.workdir, to)
        except Exception as e:
            return False, f"Ошибка запуска: {e}"
        code, out = res
        if len(out) > MAX_TOOL_RESULT:
            half = MAX_TOOL_RESULT // 2
            out = out[:half] + "\n… [вывод обрезан] …\n" + out[-half:]
        return code == 0, f"exit code: {code}\n{out.strip() if out.strip() else '(пустой вывод)'}"

    def t_web(self, url: str = "", **_: Any) -> Tuple[bool, str]:
        if not re.match(r"^https?://", url or ""):
            return False, "Нужен http(s) URL."
        try:
            raw = _Net.request(url, headers={"User-Agent": USER_AGENT, **hf_headers()}, timeout=60)
            with raw as r:
                ctype = r.headers.get("Content-Type", "")
                enc = r.headers.get_content_charset() or "utf-8"
                data = r.read(3_000_000)
        except Exception as e:
            return False, f"Не удалось скачать {url}: {e}"
        try:
            text = data.decode(enc, errors="replace")
        except LookupError:
            # неизвестное имя кодировки в заголовке — пробуем cp1251, затем utf-8
            try:
                text = data.decode("cp1251")
            except Exception:
                text = data.decode("utf-8", errors="replace")
        if "html" in ctype.lower() or text.lstrip().startswith("<"):
            text = html_to_text(text)
        if len(text) > 60_000:
            text = text[:60_000] + "\n… [обрезано]"
        return True, text

    def t_todo(self, todos: Any = None, **_: Any) -> Tuple[bool, str]:
        if not isinstance(todos, list):
            return False, "todos должен быть массивом объектов {content, status}."
        clean = []
        for t in todos:
            if isinstance(t, dict):
                content = str(t.get("content") or t.get("task") or "").strip()
                status = str(t.get("status") or "pending").strip()
                if content:
                    clean.append({"content": content, "status": status})
            elif isinstance(t, str):
                clean.append({"content": t, "status": "pending"})
        self.todos = clean
        self.on_event("__todos__")
        return True, "Список задач обновлён."


# --------------------------------------------------------------------------- #
#                              РАЗБОР ПАТЧА (unified diff)
# --------------------------------------------------------------------------- #


@dataclass
class _Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: List[Tuple[str, str]]


@dataclass
class _FilePatch:
    path: str
    is_new: bool
    hunks: List[_Hunk]
    old_lines: str = ""
    new_lines: str = ""


def _strip_diff_prefix(p: str) -> Tuple[str, bool]:
    p = p.strip().split("\t")[0]
    is_new = p.startswith("/dev/null")
    if p in ("/dev/null", ""):
        return "", is_new
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    return p.strip(), is_new


def parse_patch(patch: str) -> List[_FilePatch]:
    lines = patch.replace("\r\n", "\n").split("\n")
    files: List[_FilePatch] = []
    cur: Optional[_FilePatch] = None
    cur_h: Optional[_Hunk] = None
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            old_p, _ = _strip_diff_prefix(ln[4:])
            new_p, _ = _strip_diff_prefix(lines[i + 1][4:])
            path = new_p or old_p
            cur = _FilePatch(path=path, is_new=(not old_p), hunks=[])
            files.append(cur)
            cur_h = None
            i += 2
            continue
        m = re.match(r"^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@", ln)
        if m and cur is not None:
            os_, ol, ns, nl = (int(m.group(1)), int(m.group(2) or 1),
                               int(m.group(3)), int(m.group(4) or 1))
            cur_h = _Hunk(os_, ol, ns, nl, [])
            cur.hunks.append(cur_h)
            i += 1
            continue
        if cur_h is not None:
            if ln.startswith("\\\\"):
                i += 1
                continue
            if ln.startswith("+"):
                cur_h.lines.append(("+", ln[1:]))
            elif ln.startswith("-"):
                cur_h.lines.append(("-", ln[1:]))
            elif ln.startswith(" ") or ln == "":
                cur_h.lines.append((" ", ln[1:] if ln else ""))
            elif not ln.startswith(("diff ", "index ")):
                cur_h = None
        i += 1
    return [f for f in files if f.path and f.hunks]


def _apply_hunks(text: str, hunks: List[_Hunk], is_new: bool = False) -> Tuple[Optional[str], str]:
    if is_new and text == "":
        out_lines: List[str] = []
        for h in hunks:
            out_lines.extend(t for op, t in h.lines if op in (" ", "+"))
        return "\n".join(out_lines), "ok"
    src = text.split("\n") if text != "" else []
    out: List[str] = []
    cursor = 0
    for h in hunks:
        old = [t for op, t in h.lines if op in (" ", "-")]
        new = [t for op, t in h.lines if op in (" ", "+")]
        guess = max(0, h.old_start - 1)
        idx = _find_seq(src, old, guess, cursor)
        if idx < 0:
            # допускаем неверный номер строки — ищем по всему файлу
            idx = _find_seq(src, old, cursor, cursor)
        if idx < 0:
            return None, f"контекст не найден (около строки {h.old_start})"
        if idx < cursor:
            idx = cursor
        out.extend(src[cursor:idx])
        out.extend(new)
        cursor = idx + len(old)
    out.extend(src[cursor:])
    return "\n".join(out), "ok"


def _find_seq(src: List[str], seq: List[str], guess: int, lo: int) -> int:
    if not seq:
        return max(lo, min(guess, len(src)))
    def norm(s: str) -> str:
        return s.rstrip()
    want = [norm(s) for s in seq]
    n = len(seq)
    order = [guess]
    for d in range(1, len(src) + 1):
        if guess - d >= lo:
            order.append(guess - d)
        if guess + d + n <= len(src):
            order.append(guess + d)
        if d > 2000:
            break
    for start in order:
        if start < lo or start + n > len(src):
            continue
        if all(norm(src[start + k]) == want[k] for k in range(n)):
            return start
    return -1


def _match_mask(f: Path, include: str, base: Path) -> bool:
    """Маска файлов для grep: «*.py», «**/*.py», «src/**/*.py» — по полному и по локальному пути."""
    pat = include.replace("\\", "/").strip()
    if not pat:
        return True
    name = f.name
    candidates = [str(f).replace("\\", "/"), name]
    try:
        candidates.append(str(f.relative_to(base)).replace("\\", "/"))
    except ValueError:
        pass
    if "/" not in pat:
        return fnmatch.fnmatch(name.lower(), pat.lower())
    fpat = fnmatch.translate(pat.lower())
    rx = re.compile(fpat)
    for cand in candidates:
        c = cand.lower()
        if rx.match(c) or rx.match("./" + c) or any(
                rx.match(part) for part in (c, c.split("/", 1)[-1])):
            return True
    # последний шанс: совпадение по суффиксу пути
    suffix = pat.split("/", 1)[-1] if "**" not in pat else pat
    return fnmatch.fnmatch(name.lower(), suffix.lower())


def html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style|noscript|svg)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)<br\s*/?>", "\n", html)
    html = re.sub(r"(?i)</(p|div|li|tr|h[1-6]|section|article|pre)>", "\n", html)
    html = re.sub(r"<[^>]+>", " ", html)
    import html as _html
    text = _html.unescape(html)
    text = re.sub(r"[ \t\xa0]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


# --------------------------------------------------------------------------- #
#                                  ОБОЛОЧКА
# --------------------------------------------------------------------------- #


def shell_argv(cmd: str) -> List[str]:
    if IS_WINDOWS:
        for cand in ("pwsh", "powershell", "cmd"):
            if shutil.which(cand):
                if cand == "cmd":
                    return ["cmd", "/c", cmd]
                return [cand, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", cmd]
        return ["cmd", "/c", cmd]
    return [os.environ.get("SHELL") or "/bin/sh", "-lc", cmd]


def run_shell_command(cmd: str, cwd: Path, timeout: int = SHELL_TIMEOUT) -> Tuple[int, str]:
    p = subprocess.Popen(shell_argv(cmd), cwd=str(cwd), stdin=subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        out, _ = p.communicate(timeout=timeout)
        code = p.returncode
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
        code = -9
        out = (out or b"") + f"\n[таймаут {timeout} с — процесс убит]".encode()
    return code, (out or b"").decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
#                          РАЗБОР ВЫЗОВОВ ИНСТРУМЕНТОВ
# --------------------------------------------------------------------------- #

TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S | re.I)
TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>(.*)$", re.S | re.I)


@dataclass
class ParsedCall:
    name: str
    args: Dict[str, Any]
    error: str = ""


def _parse_tool_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return {}
        try:
            v = json.loads(s)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
        # пытаемся вытащить JSON-объект из текста
        m = re.search(r"\{.*\}", s, re.S)
        if m:
            try:
                v = json.loads(m.group(0))
                if isinstance(v, dict):
                    return v
            except json.JSONDecodeError:
                pass
        raise ValueError("аргументы не являются JSON-объектом")
    raise ValueError(f"неподдерживаемый тип аргументов {type(raw).__name__}")


def parse_tool_calls(text: str, valid_names: Sequence[str]) -> Tuple[str, List[ParsedCall]]:
    calls: List[ParsedCall] = []
    valid = set(valid_names)

    for m in TOOL_CALL_RE.finditer(text):
        body = m.group(1).strip()
        if not body:
            continue
        obj: Any = None
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            mm = re.search(r"\{.*\}", body, re.S)
            if mm:
                try:
                    obj = json.loads(mm.group(0))
                except json.JSONDecodeError:
                    obj = None
        if obj is None:
            calls.append(ParsedCall("", {}, f"Не разобрал JSON: {_clip(body, 300)}"))
            continue
        items = obj if isinstance(obj, list) else [obj]
        for it in items:
            if not isinstance(it, dict):
                calls.append(ParsedCall("", {}, "Элемент вызова не объект."))
                continue
            name = str(it.get("name") or it.get("tool") or it.get("tool_name") or "").strip()
            raw_args = (it.get("arguments") if "arguments" in it
                        else it.get("parameters", it.get("args", it.get("input", {}))))
            try:
                args = _parse_tool_args(raw_args)
            except ValueError as e:
                calls.append(ParsedCall(name, {}, f"Аргументы: {e}"))
                continue
            calls.append(ParsedCall(name, args))
    clean = TOOL_CALL_RE.sub("", text).strip()

    # вызов без закрывающего тега <tool_call> (частая проблема локальных моделей)
    if not calls:
        m = TOOL_CALL_OPEN_RE.search(text)
        if m:
            body = m.group(1)
            calls, end_rel = _recover_open_call(body, valid)
            if end_rel >= 0:
                clean = (text[:m.start()] + body[end_rel:]).strip()
        if not calls:
            m = re.search(r"```(?:json|tool_call|xml)?\s*(\{\s*[\"']?(?:name|tool)[\"']?\s*:.*?)\s*```",
                          text, re.S)
            if m:
                try:
                    obj = json.loads(m.group(1).strip())
                    if isinstance(obj, dict) and (obj.get("name") or obj.get("tool")):
                        calls.append(ParsedCall(str(obj.get("name") or obj.get("tool")),
                                                _parse_tool_args(obj.get("arguments", {}))))
                        clean = (text[:m.start()] + text[m.end():]).strip()
                except json.JSONDecodeError:
                    pass
    for c in calls:
        if not c.error and c.name not in valid:
            c.error = (f"Неизвестный инструмент '{c.name}'. Доступны: {', '.join(sorted(valid))}.")
    return clean, calls


def _recover_open_call(body: str, valid: Sequence[str]) -> Tuple[List[ParsedCall], int]:
    """Ищет в тексте после <tool_call> JSON-объекты и возвращает (вызовы, смещение конца)."""
    calls: List[ParsedCall] = []
    dec = json.JSONDecoder()
    pos = body.find("{")
    if pos < 0:
        return calls, -1
    first_start = pos
    while pos < len(body):
        while pos < len(body) and body[pos] not in "{[":
            pos += 1
        if pos >= len(body):
            break
        try:
            obj, end = dec.raw_decode(body, pos)
        except json.JSONDecodeError:
            break
        if isinstance(obj, dict) and (obj.get("name") or obj.get("tool")):
            name = str(obj.get("name") or obj.get("tool"))
            raw_args = (obj.get("arguments") if "arguments" in obj
                        else obj.get("parameters", obj.get("args", {})))
            try:
                calls.append(ParsedCall(name, _parse_tool_args(raw_args)))
            except ValueError as e:
                calls.append(ParsedCall(name, {}, f"Аргументы: {e}"))
        else:
            break
        pos = end
        nxt = body.find("{", pos)
        if nxt < 0:
            break
        if body[pos:nxt].strip() not in ("", ","):
            break
        pos = nxt
    if not calls:
        return calls, -1
    return calls, len(body) - first_start


# --------------------------------------------------------------------------- #
#                                 ЧАТ / UI
# --------------------------------------------------------------------------- #


class StreamPrinter:
    """Печатает поток токенов; блоки <tool_call> показываются цветом вживую."""

    OPEN = "<tool_call>"
    CLOSE = "</tool_call>"

    def __init__(self) -> None:
        self.buf = ""
        self.capturing = False
        self.raw: List[str] = []
        self.tty = _live_terminal()
        self.started = False

    def feed(self, token: str) -> None:
        self.raw.append(token)
        self.buf += token
        while True:
            low = self.buf.lower()
            if not self.capturing:
                pos = low.find(self.OPEN)
                if pos >= 0:
                    self._emit(self.buf[:pos])
                    self.buf = self.buf[pos + len(self.OPEN):]
                    self.capturing = True
                    self._newline()
                    sys.stdout.write(f"{C.MAGENTA}⏵ ")
                    sys.stdout.flush()
                    continue
                # придерживаем хвост, который может оказаться началом тега
                hold = 0
                for k in range(min(len(self.buf), len(self.OPEN) - 1), 0, -1):
                    if self.OPEN[:k] == low[-k:]:
                        hold = k
                        break
                if hold:
                    if len(self.buf) > hold:
                        self._emit(self.buf[:-hold])
                        self.buf = self.buf[-hold:]
                else:
                    self._emit(self.buf)
                    self.buf = ""
                return
            pos = low.find(self.CLOSE)
            if pos >= 0:
                self._emit_body(self.buf[:pos])
                self.buf = self.buf[pos + len(self.CLOSE):]
                self.capturing = False
                self._newline()
                continue
            safe = max(0, len(self.buf) - (len(self.CLOSE) - 2))
            if safe:
                self._emit_body(self.buf[:safe])
                self.buf = self.buf[safe:]
            return

    def _newline(self) -> None:
        sys.stdout.write(C.RESET + "\n" if self.tty else "\n")
        sys.stdout.flush()

    def _emit(self, s: str) -> None:
        if s:
            self.started = True
            sys.stdout.write(s)
            sys.stdout.flush()

    def _emit_body(self, s: str) -> None:
        if not s:
            return
        s = s.replace("\n", " ")
        if self.tty:
            sys.stdout.write(f"{C.MAGENTA}{s}{C.RESET}")
        else:
            sys.stdout.write(_clip(s, 200))
        sys.stdout.flush()

    def finish(self) -> str:
        if self.buf:
            if self.capturing:
                self._emit_body(self.buf)
            else:
                self._emit(self.buf)
        self.buf = ""
        if self.capturing:
            self._newline()
        elif self.started:
            sys.stdout.write(C.RESET + "\n" if self.tty else "\n")
            sys.stdout.flush()
        else:
            if self.tty:
                sys.stdout.write(C.RESET)
        return "".join(self.raw)


HELP_TEXT = f"""
{C.BOLD}Команды:{C.RESET}
  {C.CYAN}/help{C.RESET}            эта справка
  {C.CYAN}/tools{C.RESET}           список инструментов агента
  {C.CYAN}/undo{C.RESET}            откатить последнюю группу правок файлов
  {C.CYAN}/history{C.RESET}         сколько сообщений в контексте
  {C.CYAN}/files{C.RESET}           файлы в рабочей папке
  {C.CYAN}/cwd <путь>{C.RESET}      сменить рабочую папку агента
  {C.CYAN}/yes{C.RESET} | {C.CYAN}/ask{C.RESET}   разрешать правки/команды без вопросов / с вопросами
  {C.CYAN}/new{C.RESET}             начать новый диалог (контекст очищается)
  {C.CYAN}/clear{C.RESET}           очистить экран
  {C.CYAN}/stats{C.RESET}           статистика последнего ответа
  {C.CYAN}/server{C.RESET}          адрес API llama-server, лог
  {C.CYAN}/mascot{C.RESET}          показать {MASCOT_NAME} и его анимации
  {C.CYAN}/exit{C.RESET} | {C.CYAN}/quit{C.RESET}    выход
Multi-line ввод: заканчивайте строку символом {C.BOLD}\\{C.RESET} или пишите в тройных кавычках.
Приглашение: {C.BOLD}>{C.RESET} обычный ввод, {C.BOLD}!{C.RESET} — выполнить shell-команду вручную.
"""


class ChatSession:
    def __init__(self, client: ChatClient, tools: AgentTools, cfg: Dict[str, Any],
                 server: ServerProcess, model_path: Path) -> None:
        self.client = client
        self.tools = tools
        self.cfg = cfg
        self.server = server
        self.model_path = model_path
        self.messages: List[Dict[str, Any]] = []
        self.last_stats: Dict[str, Any] = {}
        self._trimmed_notice = False
        self._sys_prompt = self._build_system_prompt()
        self.tools.new_turn()

    def _build_system_prompt(self) -> str:
        names = ", ".join(t.name for t in self.tools.tools.values())
        env = (f"\nТекущая рабочая папка: {self.tools.workdir}\n"
               f"Платформа: {platform.system()} {platform.machine()}, оболочка: "
               f"{'PowerShell' if IS_WINDOWS else os.environ.get('SHELL', '/bin/sh')}\n"
               f"Режим разрешений: {self.tools._write_policy()} (правки файлов и команды "
               f"оболочки {'подтверждаются пользователем' if self.tools.approve else 'выполняются сразу'}).")
        return SYSTEM_PROMPT.replace("{tools}", names) + env

    # ------------------------------------------------------------------ UI #
    def _ask_approval(self, desc: str, body: str) -> bool:
        """Спрашивает разрешение: y — да, a — да и больше не спрашивать, n — нет."""
        lines = (body or "").splitlines()
        print()
        print(f"{C.BYELLOW}┌─ требуется разрешение ─────────────────────────────{C.RESET}")
        print(f"{C.BYELLOW}│{C.RESET} {C.BOLD}{desc}{C.RESET}")
        for ln in lines[:25]:
            print(f"{C.BYELLOW}│{C.RESET} {_clip(ln, 160)}")
        if len(lines) > 25:
            print(f"{C.BYELLOW}│{C.RESET} {C.GREY}… ещё {len(lines) - 25} строк{C.RESET}")
        print(f"{C.BYELLOW}└────────────────────────────────────────────────────{C.RESET}")
        while True:
            try:
                ans = input(f"{C.BGREEN}y{C.RESET}es / {C.BOLD}a{C.RESET}ll / "
                            f"{C.BRED}n{C.RESET}o  [y]› ").strip().lower()
            except EOFError:
                return False
            except KeyboardInterrupt:
                print()
                return False
            if ans in ("a", "all", "always", "в", "всё", "все"):
                self.tools.approve = False
                note("Дальше правки и команды выполняются без подтверждений.")
                try:
                    self._sys_prompt = self._build_system_prompt()
                except Exception:
                    pass
                return True
            if ans in ("", "y", "yes", "д", "да", "1"):
                return True
            if ans in ("n", "no", "н", "нет", "0"):
                return False

    def _todos_event(self) -> None:
        if not self.tools.todos:
            return
        icons = {"completed": f"{C.BGREEN}✔{C.RESET}", "in_progress": f"{C.BYELLOW}▶{C.RESET}",
                 "pending": f"{C.GREY}○{C.RESET}"}
        print(f"{C.BOLD}  задачи:{C.RESET}")
        for t in self.tools.todos:
            st = t.get("status", "pending")
            print(f"    {icons.get(st, '•')} {t['content']}")

    def _run_call(self, call: ParsedCall) -> Tuple[bool, str]:
        if call.error:
            return False, "Ошибка вызова: " + call.error
        tool = self.tools.tools.get(call.name)
        if not tool:
            return False, f"Неизвестный инструмент: {call.name}"
        try:
            return tool.run(**call.args)
        except TypeError as e:
            return False, f"Неверные аргументы для {call.name}: {e}"
        except Exception as e:  # noqa: BLE001 — инструмент не должен ронять агент
            return False, f"{type(e).__name__}: {e}"

    # --------------------------------------------------------------- ходы #
    def send(self, user_text: str) -> None:
        try:
            self._send_loop(user_text)
        finally:
            self.tools.journal.flush()

    def _send_loop(self, user_text: str) -> None:
        self.tools.new_turn()
        self.messages.append({"role": "user", "content": user_text})
        self._trim_history()
        step = 0
        while True:
            step += 1
            if step > MAX_AGENT_STEPS:
                warn(f"Достигнут лимит {MAX_AGENT_STEPS} шагов агента. /new — начать заново.")
                return
            payload = [{"role": "system", "content": self._sys_prompt}] + self.messages
            printer = StreamPrinter()
            extra: Dict[str, Any] = {}
            text = ""
            try:
                with MascotAnim("модель думает") as spin:
                    first = {"done": False}

                    def on_token(tok: str) -> None:
                        if not first["done"]:
                            first["done"] = True
                            spin.stop()
                            if _live_terminal():
                                sys.stdout.write(f"{C.BMAGENTA}{Mascot.line('think')}{C.RESET} ")
                                sys.stdout.flush()
                        printer.feed(tok)

                    try:
                        text, extra, _ = self.client.stream(payload, on_token)
                    except KeyboardInterrupt:
                        self.client.cancel()
                        raise
                    except RuntimeError as e:
                        printer.finish()
                        raise
            except KeyboardInterrupt:
                print()
                warn("Прервано пользователем.")
                self.messages.append({"role": "assistant", "content": "(ответ прерван)"})
                return
            except RuntimeError as e:
                err(str(e))
                self._maybe_recover(e)
                return
            text = printer.finish()
            self.last_stats = {
                "prompt_tokens": extra.get("prompt_tokens"),
                "completion_tokens": extra.get("completion_tokens"),
                "tok_per_s": extra.get("tok_per_s"),
                "first_token_ts": extra.get("first_token_ts"),
                "steps": step,
            }
            clean, calls = parse_tool_calls(text, list(self.tools.tools.keys()))
            if not calls:
                self.messages.append({"role": "assistant", "content": clean if clean else text})
                self._trim_history()
                Mascot.say("готово", "happy")
                return
            # историю пополняем «как есть»: модель должна видеть свои вызовы
            self.messages.append({"role": "assistant",
                                  "content": text if text.strip() else "(вызов инструментов)"})
            for call in calls:
                label = call.name or "?"
                args_brief = self._brief_args(call)
                face = "error" if call.error else "works"
                print(f"{C.MAGENTA}{Mascot.line(face)} ⏺ {label}{C.RESET} "
                      f"{C.GREY}{args_brief}{C.RESET}" if _live_terminal()
                      else f"{C.MAGENTA}⏺ {label}{C.RESET} {C.GREY}{args_brief}{C.RESET}")
                t0 = time.time()
                self.tools.on_event = (lambda m: self._todos_event()
                                       if m == "__todos__" else None)
                try:
                    with MascotAnim(f"лапы делают {label}", working=True):
                        success, result = self._run_call(call)
                except KeyboardInterrupt:
                    print()
                    warn("Прервано во время выполнения инструмента.")
                    return
                dt = time.time() - t0
                self._print_result(success, result, dt)
                self.messages.append({
                    "role": "tool",
                    "name": label,
                    "content": f"[{label}: {'OK' if success else 'ОШИБКА'}, {dt:.2f}с]\n{result}",
                })
            self._trim_history()

    @staticmethod
    def _brief_args(call: ParsedCall) -> str:
        a = call.args or {}
        for k in ("command", "path", "url", "pattern"):
            if k in a:
                v = str(a[k])
                extra = f" «{a.get('old_string', '')[:60]}…»" if k == "path" and "old_string" in a else ""
                return _clip(v, 120) + extra
        return _clip(json.dumps(a, ensure_ascii=False), 100) if a else ""

    @staticmethod
    def _print_result(success: bool, result: str, dt: float) -> None:
        lines = (result or "").splitlines()
        max_lines = 12 if _live_terminal() else 60
        if success:
            face = "happy"
            tag = f"{C.BGREEN}✔{C.RESET}"
        else:
            face = "error"
            tag = f"{C.BRED}✖{C.RESET}"
        mascot = f"{C.BMAGENTA}{Mascot.line(face)}{C.RESET} " if _live_terminal() else ""
        print(f"  {mascot}{tag} {C.GREY}{dt:.2f}s{C.RESET}")
        for ln in lines[:max_lines]:
            print(f"  {C.GREY}│ {_clip(ln, 160)}{C.RESET}")
        if len(lines) > max_lines:
            print(f"  {C.GREY}│ … ещё {len(lines) - max_lines} строк{C.RESET}")

    def _maybe_recover(self, e: Exception) -> None:
        msg = str(e).lower()
        if "контекст" in msg or "context" in msg or " 400" in msg or "http 400" in msg:
            warn("Похоже, контекст модели переполнен: /new — новый диалог, "
                 "или запустите с большим --ctx.")

    def _trim_history(self) -> None:
        if len(self.messages) > MAX_HISTORY_MESSAGES:
            cut = len(self.messages) - MAX_HISTORY_MESSAGES
            while cut < len(self.messages) and self.messages[cut].get("role") == "tool":
                cut += 1
            self.messages = self.messages[cut:]
            if not self._trimmed_notice:
                self._trimmed_notice = True
                note(f"Контекст сокращён до последних {len(self.messages)} сообщений "
                     f"(/new — начать новый диалог).")

    # --------------------------------------------------------- slash-команды #
    def handle_slash(self, line: str) -> None:
        parts = line.strip().split()
        cmd = parts[0].lower()
        arg = line.strip()[len(cmd):].strip()
        if cmd in ("/exit", "/quit", "/q"):
            raise SystemExit(0)
        if cmd not in ("/help", "/h", "/?", "/tools", "/undo", "/history", "/files",
                       "/cwd", "/yes", "/ask", "/new", "/clear", "/stats", "/server",
                       "/mascot"):
            warn(f"Неизвестная команда {cmd}. /help")
            return
        if cmd in ("/help", "/h", "/?"):
            Mascot.draw_help()
            print(HELP_TEXT)
        elif cmd == "/tools":
            print(f"{C.BOLD}Инструменты агента:{C.RESET}")
            for t in self.tools.tools.values():
                mark = f"{C.BYELLOW}(пишет){C.RESET}" if t.write else ""
                print(f"  {C.MAGENTA}{t.name:<12}{C.RESET} {mark} {t.description}")
        elif cmd == "/undo":
            self.tools.journal.flush()
            n = self.tools.journal.undo_last()
            if n:
                ok(f"Откачено изменений: {n}")
            else:
                note("Нечего откатывать.")
        elif cmd == "/history":
            roles: Dict[str, int] = {}
            for m in self.messages:
                roles[m["role"]] = roles.get(m["role"], 0) + 1
            print("  " + ", ".join(f"{k}: {v}" for k, v in roles.items()) or "пусто")
        elif cmd == "/files":
            success, res = self.tools.t_list(".")
            print(res)
        elif cmd == "/cwd":
            if not arg:
                print(f"  {self.tools.workdir}")
            else:
                p = Path(arg).expanduser().resolve()
                if p.is_dir():
                    self.tools.workdir = p
                    self._sys_prompt = self._build_system_prompt()
                    ok(f"Рабочая папка: {p}")
                else:
                    err("Папка не найдена.")
        elif cmd == "/yes":
            self.tools.approve = False
            self.tools.readonly = False
            self.cfg["approve"] = False
            self._sys_prompt = self._build_system_prompt()
            ok("Правки и команды выполняются без подтверждений (в этой сессии).")
        elif cmd == "/ask":
            self.tools.approve = True
            self.tools.readonly = False
            self.cfg["approve"] = True
            self._sys_prompt = self._build_system_prompt()
            ok("Правки и команды будут подтверждаться.")
        elif cmd == "/new":
            self.messages.clear()
            self.tools.todos = []
            self.tools.new_turn()
            ok("Новый диалог.")
        elif cmd == "/clear":
            os.system("cls" if IS_WINDOWS else "clear")
        elif cmd == "/stats":
            s = self.last_stats or {}
            print(f"  токенов промпта: {s.get('prompt_tokens')}, "
                  f"сгенерировано: {s.get('completion_tokens')}, "
                  f"скорость: {s.get('tok_per_s') and round(s['tok_per_s'], 1)} т/с, "
                  f"шагов агента: {s.get('steps')}")
        elif cmd == "/server":
            print(f"  API: {self.server.base}")
            print(f"  модель: {self.model_path}")
            print(f"  лог: {self.server.log_path}")
        elif cmd == "/mascot":
            Mascot.draw_help()
            Mascot.wave()
            Mascot.run()
            Mascot.jump()
            note(f"{MASCOT_NAME}: мигаю и барабаню лапами, когда модель думает, "
                 f"радуюсь успешным правкам и грущу, если что-то сломалось.")


# --------------------------------------------------------------------------- #
#                                   ГЛАВНОЕ
# --------------------------------------------------------------------------- #


def _read_multiline(first: str) -> str:
    lines = []
    line = first
    in_fence = first.startswith('"""') or first.startswith("'''")
    fence = first[:3] if in_fence else ""
    if in_fence:
        rest = first[3:]
        if rest.endswith(fence) and len(rest) >= 3:
            return rest[:-3]
        lines.append(rest)
    while True:
        try:
            nxt = input(f"{C.GREY}… {C.RESET}")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if in_fence:
            if nxt.strip().endswith(fence):
                lines.append(nxt.strip()[: -len(fence)])
                break
            lines.append(nxt)
        else:
            if line.endswith("\\"):
                lines[-1] = lines[-1][:-1] if lines else ""
                lines.append(nxt)
                line = nxt
                continue
            lines.append(nxt)
            break
    return "\n".join(lines)


def repl(session: ChatSession) -> None:
    print()
    print(f"{C.BOLD}Готов к работе.{C.RESET} Напишите задачу — агент сам прочитает файлы, "
          f"создаст и изменит их.")
    print(f"{C.GREY}/help — команды, /undo — откат правок, /mascot — {MASCOT_NAME}, "
          f"/exit — выход. Enter на пустой строке — многострочный ввод.{C.RESET}")
    w = session.tools.workdir
    greetings = re.compile(r"^(привет|прив|здаров|здравствуй|хай|hi|hello|hey|yo)\b", re.I)
    while True:
        try:
            prompt = f"{C.BGREEN}{C.BOLD}{w.name if w.name else w}{C.RESET} {C.BCYAN}›{C.RESET} "
            line = input(prompt)
        except (EOFError, KeyboardInterrupt):
            print()
            Mascot.sleep(1.0)
            print(f"  {C.BMAGENTA}{Mascot.line('sleep')}{C.RESET} Пока-пока!")
            return
        if not line.strip():
            continue
        if line.startswith("/"):
            try:
                session.handle_slash(line)
            except SystemExit:
                raise
            continue
        if line.startswith("!"):
            cmd = line[1:].strip()
            if cmd:
                code, out = run_shell_command(cmd, session.tools.workdir)
                print(out.rstrip())
                if code:
                    warn(f"exit code {code}")
            continue
        if greetings.match(line.strip()):
            Mascot.wave()
        text = _read_multiline(line)
        if not text.strip():
            continue
        print()
        try:
            session.send(text)
        except SystemExit:
            raise
        except KeyboardInterrupt:
            print()
            warn("Прервано.")
        print()


def pick_model_flow(P: Dict[str, Path], cfg: Dict[str, Any], auto_yes: bool,
                    model_arg: str = "") -> Path:
    ensure_dirs(P)
    existing = sorted([p for p in P["models"].glob("*.gguf") if not p.name.endswith(".part")],
                      key=lambda p: -p.stat().st_size)
    if model_arg:
        local = Path(model_arg).expanduser()
        if local.is_file():
            return local.resolve()
        candidate = P["models"] / local.name
        if candidate.is_file():
            return candidate
        return download_model(P, cfg, model_arg)

    if auto_yes:
        if existing:
            return existing[0]
        raise SystemExit("Укажите --model <url|path>. Список уже скачанных: (пусто)")

    rule("Выбор модели")
    if existing:
        rows = [(str(i + 1), p.name, fmt_bytes(p.stat().st_size)) for i, p in enumerate(existing)]
        fmt_table(rows, ("№", "файл", "размер"))
        print(f"  {C.CYAN} d{C.RESET}) скачать ещё одну модель с Hugging Face")
        while True:
            v = ask("Номер или d", "1").lower()
            if v in ("d", "dlin", "download", "с"):
                break
            try:
                return existing[int(v) - 1]
            except (ValueError, IndexError):
                err("Нет такого номера.")
    print(f"  Вставьте ссылку на модель с Hugging Face.")
    print(f"  Подходит {C.BOLD}прямая ссылка на .gguf{C.RESET} или {C.BOLD}страница репозитория{C.RESET} "
          f"(тогда я покажу GGUF-файлы на выбор).")
    print(f"  {C.GREY}Пример: https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF{C.RESET}")
    while True:
        raw = ask("Ссылка на модель")
        if not raw:
            err("Пустая ссылка.")
            continue
        try:
            return download_model(P, cfg, raw)
        except (KeyboardInterrupt, EOFError):
            raise
        except Exception as e:
            err(str(e))
            if not ask_yes_no("Попробовать другую ссылку?", True):
                raise SystemExit(1)


def download_model(P: Dict[str, Path], cfg: Dict[str, Any], raw: str) -> Path:
    ref = parse_hf_ref(raw)
    token = cfg.get("hf_token", "")
    ref = hf_resolve(ref, token)
    fname = model_filename_for(ref)
    dest = P["models"] / fname
    if dest.exists() and dest.stat().st_size > 0:
        ok(f"Модель уже скачана: {dest}")
        return dest
    print()
    note(f"Скачиваю {C.BOLD}{ref.repo_id or ref.url}{C.RESET}")
    if ref.file:
        print(f"  файл: {ref.file}")
    note(f"В папку: {dest.parent}")
    download(ref.url, dest, label=fname[:22], headers=hf_headers(token))
    try:
        P["last_model"].write_text(str(dest), encoding="utf-8")
    except Exception:
        pass
    return dest


def check_disk_space(P: Dict[str, Path], need_bytes: int) -> None:
    try:
        free = shutil.disk_usage(P["home"]).free
        if free < need_bytes * 1.1:
            warn(f"Мало места на диске: свободно {fmt_bytes(free)}, нужно ≈{fmt_bytes(need_bytes)}.")
    except Exception:
        pass


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="local_agent.py",
        description="Локальный движок LLM (llama.cpp) + ИИ-агент в стиле Claude Code. "
                    "После первого запуска программа доступна командой stai.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Примеры:\n"
               "  stai\n"
               "  stai --model https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF\n"
               "  stai --cwd C:/projects/app --yes --gpu-layers 99\n"
               "  stai --home D:/ai-check            # сменить общую папку программы\n"
               "  stai --set-home                    # заново выбрать папку и пересоздать настройку\n")
    ap.add_argument("--model", default="", help="URL Hugging Face или путь к .gguf")
    ap.add_argument("--cwd", default="", help="рабочая папка агента (по умолчанию текущая)")
    ap.add_argument("--yes", action="store_true", help="не спрашивать разрешений/подтверждений")
    ap.add_argument("--ctx", type=int, default=None, help="размер контекста")
    ap.add_argument("--gpu-layers", type=int, default=None, help="слоёв на GPU (0 = CPU)")
    ap.add_argument("--threads", type=int, default=None, help="потоков CPU")
    ap.add_argument("--port", type=int, default=None, help="порт llama-server")
    ap.add_argument("--token", default="", help="HF-токен для gated-моделей")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--server-only", action="store_true", help="только поднять API, без чата")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--no-mascot", action="store_true", help="отключить маскота и анимации")
    ap.add_argument("--home", default="", metavar="ПАПКА",
                    help="единая папка программы (запоминается)")
    ap.add_argument("--set-home", action="store_true",
                    help="заново выбрать папку программы и выполнить настройку")
    ap.add_argument("--no-bootstrap", action="store_true",
                    help="не ставить зависимости и не прописывать команду stai")
    ap.add_argument("--version", action="version", version=f"{APP_NAME} {VERSION}")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass

    # цвета и маскот включаем ДО первого вывода, иначе --no-color не действует
    enable_vt_mode()
    if args.no_color or not _live_terminal() or os.environ.get("NO_COLOR"):
        C.enabled = False
        _colors()
    Mascot.enabled = not args.no_mascot
    Mascot.tty = bool(_live_terminal()) and Mascot.enabled

    banner()
    if Mascot.enabled:
        Mascot.intro()

    # 0) единая папка программы --------------------------------------------
    home = resolve_home()
    if args.home or args.set_home or home is None:
        try:
            home = setup_home_interactive(args.home)
        except (KeyboardInterrupt, EOFError):
            raise SystemExit(130)
        except SystemExit:
            raise
        except Exception as e:
            err(f"Папка программы: {e}")
            return 1
    P = home_paths(home)
    try:
        ensure_dirs(P)
    except RuntimeError as e:
        err(str(e))
        return 1

    # 0.5) зависимости + команда stai (один раз; возможен перезапуск в venv)
    if not args.no_bootstrap and os.environ.get(SKIP_BOOTSTRAP_ENV) != "1":
        try:
            bootstrap(P, auto=not args.no_bootstrap)
        except (KeyboardInterrupt, EOFError):
            raise SystemExit(130)
        except SystemExit:
            raise
        except Exception as e:
            warn(f"Настройка окружения прервалась ({e}) — продолжаю запуск.")

    cfg = load_config(P)
    cfg["_home"] = str(P["home"])
    if args.ctx is not None:
        cfg["ctx"] = args.ctx
    if args.gpu_layers is not None:
        cfg["gpu_layers"] = args.gpu_layers
    if args.threads is not None:
        cfg["threads"] = args.threads
    if args.port is not None:
        cfg["port"] = args.port
    if args.token:
        cfg["hf_token"] = args.token
    if args.yes:
        cfg["approve"] = False

    note(f"Папка программы: {P['home']}")
    if args.model and not re.match(r"^(https?://|[\w.-]+/[\w.-]+$)", args.model, re.I):
        warn(f"--model {args.model!r} не похоже на URL Hugging Face — если это локальный путь, "
             "используйте полный путь к файлу .gguf.")

    # 1) llama.cpp ----------------------------------------------------------
    try:
        inst = ensure_llamacpp(P, cfg, auto_yes=args.yes)
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(130)
    except Exception as e:
        err(f"llama.cpp: {e}")
        return 1

    # 2) модель -------------------------------------------------------------
    try:
        model_path = pick_model_flow(P, cfg, args.yes, args.model)
    except (KeyboardInterrupt, EOFError):
        raise SystemExit(130)
    except SystemExit:
        raise
    except Exception as e:
        err(f"Модель: {e}")
        return 1
    try:
        size = model_path.stat().st_size
        ok(f"Модель: {model_path.name} ({fmt_bytes(size)})")
    except OSError:
        err(f"Файл модели недоступен: {model_path}")
        return 1

    # 3) рабочая папка ------------------------------------------------------
    workdir = Path(args.cwd).expanduser() if args.cwd else Path.cwd()
    if not workdir.is_dir():
        err(f"Рабочая папка не найдена: {workdir}")
        return 1
    workdir = workdir.resolve()

    # 4) запуск llama-server ------------------------------------------------
    rule("Запуск llama-server")
    srv = ServerProcess(inst, model_path, cfg)
    try:
        srv.start()
    except RuntimeError as e:
        err(str(e))
        return 1
    ok(f"Процесс запущен (PID {srv.proc.pid if srv.proc else '?'}), порт {srv.port}")
    Mascot.run()          # маскот пробегает, пока грузится модель
    ok_ready = srv.wait_ready()
    if not ok_ready:
        err("Модель не загрузилась. Последние строки лога:")
        print(srv.log_tail(60))
        srv.stop()
        if args.gpu_layers is None and (cfg.get("gpu_layers", -1) != 0):
            warn("Частая причина — нехватка VRAM. Попробуйте --gpu-layers 0 (только CPU) "
                 "или меньший квант модели.")
        return 1
    ok(f"Модель загружена. API: {srv.base}/v1  (совместим с OpenAI)")
    Mascot.jump("proud")

    if args.server_only:
        note("Режим --server-only: Ctrl+C — остановить сервер.")
        try:
            while True:
                time.sleep(1)
                if srv.proc and srv.proc.poll() is not None:
                    err("llama-server завершился.")
                    return 1
        except KeyboardInterrupt:
            print()
            return 0
        finally:
            srv.stop()

    # 5) чат ----------------------------------------------------------------
    journal = UndoJournal(P["undo"])
    tools = AgentTools(workdir, journal, approve=bool(cfg.get("approve", True)))
    client = ChatClient(srv.base, cfg.get("api_key", ""), temperature=args.temperature)
    session = ChatSession(client, tools, cfg, srv, model_path)
    tools.on_approve = session._ask_approval
    rule("Чат с агентом")
    note(f"Рабочая папка: {C.BOLD}{workdir}{C.RESET}")
    note(f"Подтверждения правок: {'ДА' if tools.approve else 'нет (--yes)'} — меняется /yes, /ask")
    try:
        repl(session)
    finally:
        print()
        note("Останавливаю llama-server…")
        srv.stop()
        ok("Готово. До встречи!")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print()
        raise SystemExit(130)
