#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stai — точка входа команды `stai` для local_agent.py.

Этот файл создаётся автоматически и копируется в папку программы
(<домашняя папка>/scripts/stai.py); сам local_agent.py остаётся там, где вы его
положили. Запускается интерпретатором из venv, поэтому внешние зависимости
доступны без активации окружения.

Ручной запуск тоже возможен:  python scripts/stai.py [аргументы local_agent.py]
"""

from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve()


def _candidates() -> list:
    out = []
    for env_name in ("LOCAL_AGENT_MAIN", "LOCAL_AGENT_SCRIPT"):
        val = os.environ.get(env_name, "").strip()
        if val:
            out.append(Path(val).expanduser())

    # <home>/scripts/stai.py — точный путь к main записан при установке
    try:
        state = json.loads((HERE.parent.parent / "state.json").read_text(encoding="utf-8"))
        if isinstance(state, dict):
            saved = str(state.get("main_script") or "").strip()
            if saved:
                out.append(Path(saved).expanduser())
    except Exception:
        pass

    # поиск рядом: текущий каталог и родители папки scripts/
    bases = [Path.cwd(), HERE.parent, HERE.parent.parent,
             HERE.parent.parent.parent, HERE.parent.parent.parent.parent]
    for base in bases:
        for depth in range(4):
            out.append(base.joinpath(*([".."] * depth)) / "local_agent.py")
    return out


main_script = None
for cand in _candidates():
    try:
        if cand.is_file():
            main_script = cand.resolve()
            break
    except OSError:
        continue

if main_script is None:
    sys.stderr.write(
        "Не найден local_agent.py.\n"
        "Укажите путь: $env:LOCAL_AGENT_MAIN = 'C:\\путь\\local_agent.py' "
        "(или LOCAL_AGENT_MAIN=/путь/local_agent.py)\n"
    )
    raise SystemExit(2)

sys.argv = [str(main_script), *sys.argv[1:]]
try:
    runpy.run_path(str(main_script), run_name="__main__")
except KeyboardInterrupt:
    raise SystemExit(130)
