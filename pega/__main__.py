import argparse
import sqlite3
import sys
from contextlib import closing

from .session import SESSION_ERRORS, ChatSession, add_session_arguments
from .store import Store


HELP = """Type a message to chat.
  /good                 Learn the last reply.
  /correct <reply>      Learn your replacement for the last reply.
  /new                  Start a fresh conversation.
  /status               Show saved examples and training updates.
  /help                 Show these commands.
  /quit                 Exit."""


def progress(phase: str) -> None:
    if phase == "loading":
        print(
            "Loading SmolLM2. The first online run may download model files.",
            flush=True,
        )
    elif phase == "training":
        print("Learning from your feedback...", flush=True)


def run(session: ChatSession) -> None:
    print("Pega is ready. Type /help for commands.")
    while True:
        try:
            text = input("\nYou> ").strip()
            if not text:
                continue
            if not text.startswith("/"):
                answer = session.reply(text, progress)
                if answer:
                    print(f"Pega> {answer}")
                else:
                    print(
                        "Pega returned an empty reply. Use /correct <reply> to teach an answer."
                    )
                continue

            command, *arguments = text.split(maxsplit=1)
            if command == "/quit" and not arguments:
                break
            if command == "/help" and not arguments:
                print(HELP)
            elif command == "/new" and not arguments:
                session.reset()
                print("Started a fresh conversation.")
            elif command == "/status" and not arguments:
                count, step = session.store.status()
                print(f"Saved examples {count} | Training updates {step}")
            elif (command == "/good" and not arguments) or (
                command == "/correct" and arguments
            ):
                changed = session.learn(arguments[0] if arguments else None, progress)
                print(
                    "Learned and saved."
                    if changed
                    else "That reply is already learned."
                )
            else:
                print("Type /help for command usage.")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        except SESSION_ERRORS as error:
            print(f"Pega could not complete the request. {error}", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m pega", description="Chat with Pega and teach it replies."
    )
    add_session_arguments(parser)
    args = parser.parse_args()
    try:
        with closing(Store(args.data_dir.expanduser() / "pega.sqlite3")) as store:
            run(ChatSession(store, args.device, args.offline))
    except (OSError, sqlite3.Error) as error:
        print(f"Pega could not open its learning profile. {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
