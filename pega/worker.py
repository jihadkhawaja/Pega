"""Line-delimited JSON interface for the desktop chat."""

import argparse
import json
import sys
from contextlib import closing, redirect_stdout
from typing import TextIO

from .session import SESSION_ERRORS, ChatSession, add_session_arguments
from .store import Store


def run(session: ChatSession, output: TextIO) -> None:
    def send(kind: str, **payload: object) -> None:
        print(json.dumps({"kind": kind, **payload}), file=output, flush=True)

    def progress(phase: str) -> None:
        send("progress", phase=phase)

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Expected a chat request object.")
            action = request.get("action")
            text = request.get("text")
            if action in ("chat", "correct") and not isinstance(text, str):
                raise ValueError("Expected text for this request.")

            response = {}
            if action == "chat":
                response["answer"] = session.reply(text, progress)
            elif action in ("approve", "correct"):
                response["changed"] = session.learn(
                    text if action == "correct" else None, progress
                )
            elif action == "reset":
                session.reset()
            elif action == "reset_learning":
                session.reset_learning()
            elif action != "status":
                raise ValueError("Unknown desktop chat action.")
            count, step = session.store.status()
            send("result", examples=count, updates=step, **response)
        except SESSION_ERRORS as error:
            send("error", message=str(error))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Pega desktop model worker.")
    add_session_arguments(parser)
    args = parser.parse_args()
    output = sys.stdout
    try:
        with closing(Store(args.data_dir.expanduser() / "pega.sqlite3")) as store:
            with redirect_stdout(sys.stderr):
                run(ChatSession(store, args.device, args.offline), output)
    except SESSION_ERRORS as error:
        print(
            json.dumps({"kind": "error", "message": str(error)}),
            file=output,
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
