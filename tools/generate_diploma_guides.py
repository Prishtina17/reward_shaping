#!/usr/bin/env python3

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from typing import Iterable, List, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "MARL Reward Shaping" / "Руководство по запуску и конфигурации"

FONT_SANS = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
FONT_MONO = Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")


@dataclass(frozen=True)
class Styles:
    page_w: int = 1240
    page_h: int = 1754  # ~A4 at 150 DPI
    margin: int = 80
    leading: int = 10

    h1_size: int = 34
    h2_size: int = 26
    body_size: int = 20
    mono_size: int = 18

    line_spacing: int = 6

    fg: Tuple[int, int, int] = (20, 20, 20)
    muted: Tuple[int, int, int] = (90, 90, 90)
    border: Tuple[int, int, int] = (210, 210, 210)
    code_bg: Tuple[int, int, int] = (245, 246, 248)
    note_bg: Tuple[int, int, int] = (240, 248, 255)


def load_fonts(st: Styles):
    if not FONT_SANS.exists() or not FONT_MONO.exists():
        raise FileNotFoundError("DejaVu fonts not found in /usr/share/fonts/truetype/dejavu/")
    return {
        "h1": ImageFont.truetype(str(FONT_SANS), st.h1_size),
        "h2": ImageFont.truetype(str(FONT_SANS), st.h2_size),
        "body": ImageFont.truetype(str(FONT_SANS), st.body_size),
        "mono": ImageFont.truetype(str(FONT_MONO), st.mono_size),
        "small": ImageFont.truetype(str(FONT_SANS), max(14, st.body_size - 4)),
    }


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
    return int(x1 - x0)


def _line_height(draw: ImageDraw.ImageDraw, font: ImageFont.ImageFont) -> int:
    x0, y0, x1, y1 = draw.textbbox((0, 0), "Ag", font=font)
    return int(y1 - y0)


def wrap_to_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    # Simple greedy wrapping on spaces, stable enough for short guides.
    words = (text or "").split()
    if not words:
        return [""]
    lines: List[str] = []
    cur: List[str] = []
    for w in words:
        cand = (" ".join(cur + [w])).strip()
        if _text_width(draw, cand, font) <= max_w or not cur:
            cur.append(w)
        else:
            lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


class PageWriter:
    def __init__(self, st: Styles, fonts):
        self.st = st
        self.fonts = fonts
        self.pages: List[Image.Image] = []
        self._new_page()

    def _new_page(self):
        im = Image.new("RGB", (self.st.page_w, self.st.page_h), "white")
        self.draw = ImageDraw.Draw(im)
        self.pages.append(im)
        self.x = self.st.margin
        self.y = self.st.margin

    def _ensure_space(self, needed_h: int):
        if self.y + needed_h <= self.st.page_h - self.st.margin:
            return
        self._new_page()

    def h1(self, text: str):
        font = self.fonts["h1"]
        lh = _line_height(self.draw, font) + self.st.line_spacing
        self._ensure_space(lh * 2)
        self.draw.text((self.x, self.y), text, fill=self.st.fg, font=font)
        self.y += lh + 10
        self._rule()

    def h2(self, text: str):
        font = self.fonts["h2"]
        lh = _line_height(self.draw, font) + self.st.line_spacing
        self._ensure_space(lh * 2)
        self.draw.text((self.x, self.y), text, fill=self.st.fg, font=font)
        self.y += lh + 6

    def p(self, text: str, muted: bool = False):
        font = self.fonts["body"]
        max_w = self.st.page_w - 2 * self.st.margin
        lines = wrap_to_width(self.draw, text, font, max_w)
        lh = _line_height(self.draw, font) + self.st.line_spacing
        self._ensure_space(lh * (len(lines) + 1))
        color = self.st.muted if muted else self.st.fg
        for line in lines:
            self.draw.text((self.x, self.y), line, fill=color, font=font)
            self.y += lh
        self.y += self.st.leading

    def bullets(self, items: Sequence[str]):
        font = self.fonts["body"]
        max_w = self.st.page_w - 2 * self.st.margin - 40
        lh = _line_height(self.draw, font) + self.st.line_spacing
        for it in items:
            lines = wrap_to_width(self.draw, it, font, max_w)
            self._ensure_space(lh * (len(lines) + 1))
            self.draw.text((self.x, self.y), "\u2022", fill=self.st.fg, font=font)
            tx = self.x + 28
            for j, line in enumerate(lines):
                self.draw.text((tx, self.y), line, fill=self.st.fg, font=font)
                self.y += lh
            self.y += int(self.st.leading / 2)
        self.y += self.st.leading

    def code(self, lines: Sequence[str]):
        font = self.fonts["mono"]
        lh = _line_height(self.draw, font) + 4
        pad = 16
        max_w = self.st.page_w - 2 * self.st.margin

        # Wrap long code lines visually (hard wrap by chars).
        rendered: List[str] = []
        for line in lines:
            if not line:
                rendered.append("")
                continue
            # crude: assume mono, fit by character count from width estimate
            char_w = max(8, _text_width(self.draw, "M", font))
            max_chars = max(20, int((max_w - 2 * pad) / char_w))
            rendered.extend(wrap(line, width=max_chars, break_long_words=True, break_on_hyphens=False))

        box_h = pad * 2 + lh * len(rendered)
        self._ensure_space(box_h + 10)

        x0 = self.x
        y0 = self.y
        x1 = self.x + max_w
        y1 = self.y + box_h
        self.draw.rectangle([x0, y0, x1, y1], fill=self.st.code_bg, outline=self.st.border, width=2)
        cy = y0 + pad
        for line in rendered:
            self.draw.text((x0 + pad, cy), line, fill=self.st.fg, font=font)
            cy += lh
        self.y = y1 + 18

    def note(self, title: str, items: Sequence[str]):
        # Light-blue callout for "важно" blocks.
        font_t = self.fonts["body"]
        font_b = self.fonts["small"]
        pad = 16
        max_w = self.st.page_w - 2 * self.st.margin

        title_lines = wrap_to_width(self.draw, title, font_t, max_w - 2 * pad)
        body_lines: List[str] = []
        for it in items:
            body_lines.extend(["- " + l for l in wrap_to_width(self.draw, it, font_b, max_w - 2 * pad)])

        lh_t = _line_height(self.draw, font_t) + 4
        lh_b = _line_height(self.draw, font_b) + 4
        box_h = pad * 2 + lh_t * len(title_lines) + 8 + lh_b * len(body_lines)
        self._ensure_space(box_h + 10)

        x0 = self.x
        y0 = self.y
        x1 = self.x + max_w
        y1 = self.y + box_h
        self.draw.rectangle([x0, y0, x1, y1], fill=self.st.note_bg, outline=self.st.border, width=2)
        cy = y0 + pad
        for line in title_lines:
            self.draw.text((x0 + pad, cy), line, fill=self.st.fg, font=font_t)
            cy += lh_t
        cy += 8
        for line in body_lines:
            self.draw.text((x0 + pad, cy), line, fill=self.st.fg, font=font_b)
            cy += lh_b
        self.y = y1 + 18

    def _rule(self):
        x0 = self.x
        x1 = self.st.page_w - self.st.margin
        y = self.y
        self.draw.line([x0, y, x1, y], fill=self.st.border, width=2)
        self.y += 18


def save_pdf(pages: List[Image.Image], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    first, rest = pages[0], pages[1:]
    first.save(str(out_path), "PDF", resolution=150.0, save_all=True, append_images=rest)


def guide_121() -> Tuple[str, List[Image.Image]]:
    st = Styles()
    fonts = load_fonts(st)
    w = PageWriter(st, fonts)

    w.h1("1.2.1 Установка серверного приложения")
    w.p("В этом проекте 'серверное приложение' это обучающая часть: Python-код (PyMARL2/QMIX) и окружение SMAC/StarCraft II, которое запускает обучение и пишет метрики.")
    w.h2("Что потребуется")
    w.bullets([
        "Операционная система Linux (желательно Ubuntu 20.04+).",
        "Conda не обязательна: скрипт сам установит Miniforge (conda-forge) в домашнюю папку.",
        "Свободное место на диске: 10+ ГБ (StarCraft II и карты).",
        "Интернет на время установки (зависимости и StarCraft II скачиваются автоматически).",
    ])
    w.h2("Установка (одной командой)")
    w.p("Установка выполняется из корня репозитория одной командой:")
    w.code(["bash bootstrap.sh"])
    w.note("Скрипт bootstrap.sh выполняет", [
        "Создаст conda-окружение (Python 3.8) или использует уже существующее.",
        "Установит PyTorch (pip wheels) и Python-зависимости проекта.",
        "Скачает и установит StarCraft II 4.10 и карты SMAC.",
        "Сделает короткий проверочный запуск (smoke), чтобы убедиться, что все работает.",
    ])
    w.h2("Критерии успешной установки")
    w.bullets([
        "Появилась папка `~/StarCraftII` (это установленный StarCraft II).",
        "После smoke-запуска появились папки `results/sacred` и/или `results/tb_logs` (это логи экспериментов).",
    ])
    w.h2("Типовые причины ошибок")
    w.bullets([
        "Отсутствует доступ в интернет (не скачиваются зависимости/StarCraft II).",
        "Не установлены системные утилиты `unzip`/`wget`/`curl` (нужны для скачивания и распаковки).",
        "Недостаточно места на диске (StarCraft II и карты).",
    ])

    return "1.2.1_Установка_серверного_приложения.pdf", w.pages


def guide_122() -> Tuple[str, List[Image.Image]]:
    st = Styles()
    fonts = load_fonts(st)
    w = PageWriter(st, fonts)

    w.h1("1.2.2 Установка клиентского приложения")
    w.p("В рамках дипломной структуры 'клиентское приложение' это скрипты и конфиги, которые запускают эксперименты и показывают результаты (графики, логи).")
    w.h2("Главные файлы")
    w.bullets([
        "`bootstrap.sh` - установка и первый запуск одной командой.",
        "`run_all_shapings.sh` - полный дипломный прогон (карты x shapings).",
        "`run_tensorboard.sh` - запуск TensorBoard для просмотра графиков.",
        "`src/config/` - YAML-конфиги окружений и алгоритмов.",
    ])
    w.h2("Как запустить дипломный прогон")
    w.p("После установки одной командой можно запустить полный прогон (это долго):")
    w.code(["bash bootstrap.sh --run-mode full"])
    w.h2("Как посмотреть графики (TensorBoard)")
    w.code(["./run_tensorboard.sh", "# затем открыть URL, который выведет скрипт"])
    w.note("Где лежат логи", [
        "TensorBoard: `results/tb_logs/<unique_token>/`.",
        "Sacred артефакты: `results/sacred/<map>/<experiment_name>/`.",
    ])
    return "1.2.2_Установка_клиентского_приложения.pdf", w.pages


def guide_123() -> Tuple[str, List[Image.Image]]:
    st = Styles()
    fonts = load_fonts(st)
    w = PageWriter(st, fonts)

    w.h1("1.2.3 Настройка сервера СУБД и файла БД")
    w.p("В проекте нет отдельной СУБД (PostgreSQL/MySQL и т.п.). Настройка базы данных не требуется.")
    w.h2("Как хранится информация")
    w.bullets([
        "Конфиги и метаданные запусков: Sacred в `results/sacred/`.",
        "Графики и скаляры: TensorBoard в `results/tb_logs/`.",
        "Чекпоинты моделей: `results/models/<unique_token>/<t_env>/`.",
        "Финальные выгрузки для диплома: `final_tb_logs/` и `final_metrics/`.",
    ])
    w.note("Примечание", [
        "Роль 'информационной системы' выполняет связка Sacred + TensorBoard + файловое хранилище артефактов.",
        "Подход характерен для исследовательских ML/DRL проектов: приоритетом является воспроизводимость и трассируемость экспериментов.",
    ])
    return "1.2.3_Настройка_сервера_СУБД_и_файла_БД.pdf", w.pages


def guide_124() -> Tuple[str, List[Image.Image]]:
    st = Styles()
    fonts = load_fonts(st)
    w = PageWriter(st, fonts)

    w.h1("1.2.4 Установка и конфигурация информационной системы")
    w.p("Информационная система проекта это сбор, хранение и визуализация метрик экспериментов (win rate, shaping-диагностики, чекпоинты, артефакты).")
    w.h2("Что устанавливается автоматически")
    w.bullets([
        "Sacred FileStorageObserver: сохраняет параметры запуска и артефакты в `results/sacred/`.",
        "TensorBoard: сохраняет скаляры в `results/tb_logs/`.",
        "Скрипт генерации итоговых графиков: `plot_final_metrics.py`.",
    ])
    w.h2("Запуск и просмотр метрик")
    w.code(["./run_tensorboard.sh"])
    w.p("Готовые выгрузки для диплома лежат в `final_tb_logs/` и `final_metrics/`.")
    w.h2("Генерация итоговых графиков из final_tb_logs")
    w.code(["python plot_final_metrics.py"])
    w.h2("Что считать успехом")
    w.bullets([
        "TensorBoard открылся и показывает графики.",
        "В `final_metrics/<map>/` появились png-файлы после `plot_final_metrics.py`.",
    ])
    return "1.2.4_Установка_и_конфигурация_информационной_системы.pdf", w.pages


def guide_125() -> Tuple[str, List[Image.Image]]:
    st = Styles()
    fonts = load_fonts(st)
    w = PageWriter(st, fonts)

    w.h1("1.2.5 Установка и конфигурация мобильного приложения")
    w.p("Мобильное приложение в рамках данного проекта не реализовано.")
    w.h2("Почему раздел существует")
    w.bullets([
        "Раздел добавлен для соответствия структуре дипломного проекта.",
        "Проект предназначен для оффлайн-обучения MARL моделей на ПК/сервере.",
    ])
    w.h2("Статус компонента")
    w.p("Мобильный клиент не требуется для целей дипломного проекта: разработка и сравнение reward shaping методов, а также воспроизводимые эксперименты выполняются на ПК/сервере.")
    return "1.2.5_Установка_и_конфигурация_мобильного_приложения.pdf", w.pages


def guide_126() -> Tuple[str, List[Image.Image]]:
    st = Styles()
    fonts = load_fonts(st)
    w = PageWriter(st, fonts)

    w.h1("1.2.6 Установка и конфигурация прочих утилит")
    w.p("В проекте есть набор утилит для установки окружения и анализа результатов.")
    w.h2("Скрипты")
    w.bullets([
        "`bootstrap.sh` - основной сценарий 'одной командой' (установка + smoke).",
        "`install_dependencies.sh` - установка зависимостей и PyTorch (pip wheels).",
        "`install_sc2.sh` - StarCraft II 4.10 + карты SMAC.",
        "`install_gfootball.sh` - (опционально) Google Football окружение.",
        "`run_tensorboard.sh` - запуск TensorBoard.",
        "`plot_final_metrics.py` - генерация png-графиков из final_tb_logs.",
    ])
    w.h2("Рекомендуемый сценарий запуска")
    w.code(["bash bootstrap.sh"])
    w.note("Если нужно 'все и сразу'", [
        "Полный прогон занимает много времени и ресурсов.",
        "Запускайте его только если это требуется для демонстрации: `bash bootstrap.sh --run-mode full`.",
    ])
    return "1.2.6_Установка_и_конфигурация_прочих_утилит.pdf", w.pages


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    guides = [guide_121, guide_122, guide_123, guide_124, guide_125, guide_126]
    for fn in guides:
        name, pages = fn()
        save_pdf(pages, OUT_DIR / name)

    # Remove test artifact if present.
    test_pdf = OUT_DIR / "_ru_test.pdf"
    if test_pdf.exists():
        try:
            test_pdf.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
