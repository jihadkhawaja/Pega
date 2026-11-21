import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from pega.session import ChatSession
from pega.store import Lesson, Store, Turn


class LearningTests(unittest.TestCase):
    def setUp(self):
        folder = self.enterContext(tempfile.TemporaryDirectory())
        self.store = Store(Path(folder) / "pega.sqlite3")
        self.addCleanup(self.store.close)
        self.session = ChatSession(self.store, "cpu", True)
        self.model = Mock()
        self.session.model = self.model
        self.progress = Mock()

    def test_a_learned_greeting_does_not_override_an_unrelated_followup(self):
        greeting = Turn("Hi", "Hello, how can I help?")
        self.store.save(Lesson((), greeting), 0, b"adapter")
        self.assertEqual(self.session.reply("Hi", self.progress), greeting.answer)
        self.model.reply.assert_not_called()

        prompt = "Do you know who I am?"
        self.model.reply.return_value = "I don't know your name yet."
        self.assertEqual(
            self.session.reply(prompt, self.progress), self.model.reply.return_value
        )
        self.model.reply.assert_called_once_with((greeting,), prompt, use_adapter=False)
        self.model.load_checkpoint.assert_not_called()

    def test_a_related_question_uses_the_saved_lesson_and_adapter(self):
        example = Turn("What is my project codename?", "Violet Comet.")
        self.store.save(Lesson((), example), 0, b"adapter")
        prompt = "Tell me my project codename."
        self.model.reply.return_value = example.answer

        self.assertEqual(self.session.reply(prompt, self.progress), example.answer)
        self.model.load_checkpoint.assert_called_once_with(1, b"adapter")
        self.model.reply.assert_called_once_with((example,), prompt, use_adapter=True)

    def test_reset_prevents_an_in_flight_write_from_restoring_a_lesson(self):
        lesson = Lesson((), Turn("Hi", "Hello!"))
        self.assertTrue(self.store.save(lesson, 0, b"adapter"))
        self.session.reply("Hi", self.progress)
        step, weights = self.store.checkpoint()
        self.session.reset_learning()

        self.assertEqual(self.store.status(), (0, 0))
        self.assertEqual(self.store.checkpoint(), (step + 1, None))
        self.assertIsNone(self.store.answer_for((), "Hi"))
        self.assertIsNone(self.session.last_lesson)
        self.assertFalse(self.store.save(lesson, step, weights))
        self.assertTrue(self.store.was_reset_after(step))
