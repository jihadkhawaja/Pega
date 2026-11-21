import argparse
import os
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .store import Lesson, Store, Turn

if TYPE_CHECKING:
    from .model import LanguageModel


HISTORY_TURNS = 3
REPLAY_LIMIT = 3
SESSION_ERRORS = (ImportError, OSError, RuntimeError, ValueError, sqlite3.Error)


def default_data_dir() -> Path:
    windows_data = os.environ.get("LOCALAPPDATA")
    if windows_data:
        return Path(windows_data) / "Pega"
    return (
        Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        / "pega"
    )


def add_session_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--offline", action="store_true", help="Use cached model files only."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir(),
        help="Folder for the lessons and learned adapter.",
    )


class ChatSession:
    def __init__(self, store: Store, device: str, offline: bool):
        self.store = store
        self.device = device
        self.offline = offline
        self.model: LanguageModel | None = None
        self.history: tuple[Turn, ...] = ()
        self.last_lesson: Lesson | None = None

    def reset(self) -> None:
        self.history = ()
        self.last_lesson = None

    def reset_learning(self) -> None:
        self.store.reset_learning()
        self.reset()

    def _get_model(self, progress: Callable[[str], None]) -> "LanguageModel":
        if self.model is None:
            progress("loading")
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
            from .model import LanguageModel

            self.model = LanguageModel(self.device, self.offline)
        return self.model

    def reply(self, prompt: str, progress: Callable[[str], None]) -> str:
        self.last_lesson = None
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Enter a message to send.")
        answer = self.store.answer_for(self.history, prompt)
        if answer is None:
            model = self._get_model(progress)
            progress("responding")
            example = self.store.related_turn(prompt, self.history)
            context = self.history
            use_adapter = False
            if example is not None:
                step, weights = self.store.checkpoint()
                if weights is not None:
                    model.load_checkpoint(step, weights)
                    use_adapter = True
                    if example not in context:
                        context = (example, *context)
            answer = model.reply(context, prompt, use_adapter=use_adapter)
        self.last_lesson = Lesson(self.history, Turn(prompt, answer))
        self.history = (*self.history, self.last_lesson.turn)[-HISTORY_TURNS:]
        return answer

    def learn(self, correction: str | None, progress: Callable[[str], None]) -> bool:
        if self.last_lesson is None:
            raise ValueError("Ask a question before giving feedback.")
        history, turn = self.last_lesson.history, self.last_lesson.turn
        answer = turn.answer if correction is None else correction.strip()
        if not answer:
            raise ValueError("Enter a replacement reply before teaching.")

        lesson = Lesson(history, Turn(turn.prompt, answer))
        changed = False
        for _ in range(2):
            step, weights = self.store.checkpoint()
            if self.store.answer_for(history, turn.prompt) == answer:
                break

            model = self._get_model(progress)
            model.load_checkpoint(step, weights)
            progress("training")
            new_weights = model.train_on(
                lesson, self.store.recent_lessons(lesson, limit=REPLAY_LIMIT)
            )
            if self.store.save(lesson, step, new_weights):
                changed = True
                break
            if self.store.was_reset_after(step):
                raise RuntimeError(
                    "Learning was reset while training. Give your feedback again to save it."
                )
        else:
            raise RuntimeError(
                "Another session updated Pega. Give the feedback again to retry."
            )

        self.last_lesson = lesson
        self.history = (*history, lesson.turn)[-HISTORY_TURNS:]
        return changed
