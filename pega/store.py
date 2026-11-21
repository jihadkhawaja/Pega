import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Turn:
    prompt: str
    answer: str


@dataclass(frozen=True)
class Lesson:
    history: tuple[Turn, ...]
    turn: Turn


STOP_WORDS = frozenset(
    "a an are did do does for i in is it me my of on please tell that the "
    "this to was were what with you your".split()
)


def serialize_history(history: tuple[Turn, ...]) -> str:
    return json.dumps(
        [(turn.prompt, turn.answer) for turn in history], ensure_ascii=False
    )


def _terms(prompt: str) -> set[str]:
    return set(re.findall(r"[^\W_]+", prompt.casefold())) - STOP_WORDS


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30)
        try:
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS lessons (
                    history TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    step INTEGER NOT NULL UNIQUE,
                    PRIMARY KEY (history, prompt)
                );
                CREATE TABLE IF NOT EXISTS checkpoint (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    step INTEGER NOT NULL,
                    weights BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS learning_reset (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    step INTEGER NOT NULL
                );
            """)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def answer_for(self, history: tuple[Turn, ...], prompt: str) -> str | None:
        row = self.connection.execute(
            "SELECT answer FROM lessons WHERE history = ? AND prompt = ?",
            (serialize_history(history), prompt),
        ).fetchone()
        return row[0] if row else None

    def related_turn(self, prompt: str, history: tuple[Turn, ...] = ()) -> Turn | None:
        query = _terms(prompt)
        if len(query) < 2:
            return None

        best = None
        best_score = 0.0
        rows = self.connection.execute(
            "SELECT prompt, answer FROM lessons WHERE history IN (?, ?) ORDER BY step DESC",
            (serialize_history(()), serialize_history(history)),
        )
        for saved_prompt, answer in rows:
            candidate = _terms(saved_prompt)
            overlap = len(query & candidate)
            score = overlap / len(query | candidate)
            if overlap < 2 or score < 0.75:
                continue
            if score > best_score:
                best_score = score
                best = Turn(saved_prompt, answer)
            elif score == best_score and best is not None and best.answer != answer:
                best = None
        return best

    def checkpoint(self) -> tuple[int, bytes | None]:
        row = self.connection.execute(
            "SELECT step, weights FROM checkpoint WHERE id = 1"
        ).fetchone()
        return (row[0], row[1] or None) if row else (0, None)

    def status(self) -> tuple[int, int]:
        return self.connection.execute("""
            SELECT COUNT(*),
                COALESCE((SELECT step FROM checkpoint WHERE id = 1), 0)
                - COALESCE((SELECT step FROM learning_reset WHERE id = 1), 0)
            FROM lessons
        """).fetchone()

    def was_reset_after(self, step: int) -> bool:
        return (
            self.connection.execute(
                "SELECT 1 FROM learning_reset WHERE id = 1 AND step > ?", (step,)
            ).fetchone()
            is not None
        )

    def reset_learning(self) -> None:
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute(
                "SELECT step FROM checkpoint WHERE id = 1"
            ).fetchone()
            step = (row[0] if row else 0) + 1
            self.connection.execute("DELETE FROM lessons")
            self._save_checkpoint(step, b"")
            self.connection.execute(
                """
                INSERT INTO learning_reset (id, step) VALUES (1, ?)
                ON CONFLICT (id) DO UPDATE SET step = excluded.step
            """,
                (step,),
            )

    def recent_lessons(self, lesson: Lesson, limit: int) -> list[Lesson]:
        rows = self.connection.execute(
            """
            SELECT history, prompt, answer FROM lessons
            WHERE NOT (history = ? AND prompt = ?)
            ORDER BY step DESC
            LIMIT ?
        """,
            (serialize_history(lesson.history), lesson.turn.prompt, limit),
        )
        return [
            Lesson(
                tuple(Turn(*turn) for turn in json.loads(history)), Turn(prompt, answer)
            )
            for history, prompt, answer in rows
        ]

    def save(self, lesson: Lesson, expected_step: int, weights: bytes) -> bool:
        with self.connection:
            self.connection.execute("BEGIN IMMEDIATE")
            row = self.connection.execute(
                "SELECT step FROM checkpoint WHERE id = 1"
            ).fetchone()
            if (row[0] if row else 0) != expected_step:
                return False

            step = expected_step + 1
            self.connection.execute(
                """
                INSERT INTO lessons (history, prompt, answer, step)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (history, prompt) DO UPDATE SET
                    answer = excluded.answer, step = excluded.step
            """,
                (
                    serialize_history(lesson.history),
                    lesson.turn.prompt,
                    lesson.turn.answer,
                    step,
                ),
            )
            self._save_checkpoint(step, weights)
        return True

    def _save_checkpoint(self, step: int, weights: bytes) -> None:
        self.connection.execute(
            """
            INSERT INTO checkpoint (id, step, weights)
            VALUES (1, ?, ?)
            ON CONFLICT (id) DO UPDATE SET
                step = excluded.step, weights = excluded.weights
        """,
            (step, weights),
        )
