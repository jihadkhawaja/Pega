# Pega

Pega is a local desktop chat with a 135-million-parameter English language model. It learns when you approve or correct a reply. The app uses Avalonia and .NET 10 on Windows, macOS, and Linux. [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)

## Launch the desktop app

Install the .NET 10 SDK and Python 3.11 or newer. If you already have a working Pega Python environment, point `PEGA_PYTHON` at its interpreter and run `dotnet run --project desktop/Pega.Desktop` from the repository.

On Windows, these PowerShell commands create a fresh CPU environment and start the app. A CUDA build of PyTorch enables NVIDIA acceleration. [PyTorch installation options](https://pytorch.org/get-started/locally/)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
dotnet run --project desktop/Pega.Desktop
```

On macOS and Linux, run the following from the repository. For CPU-only Linux, you can install PyTorch from its CPU wheel index before installing the requirements.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
dotnet run --project desktop/Pega.Desktop
```

The first generated reply downloads about 270 MB of model files. Set `PEGA_OFFLINE=1` to use cached files only. You can also set `PEGA_DEVICE` to `cpu` or `cuda` and `PEGA_DATA_DIR` to a separate learning profile directory.

## Teach Pega

Press **Enter** to send a message and **Shift+Enter** for a new line. Click **Approve** to teach a reply as written. To teach a different answer, click **Correct reply**, enter the replacement, and click **Teach correction**. The **Learning profile** shows how many examples and training updates Pega has saved.

Pega fine-tunes a LoRA adapter on each new example and revisits recent lessons during training. An exact match on a question and its recent conversation reuses the saved reply. When a new prompt matches a lesson by shared keywords, the model can use the adapter and the example. Other prompts use the base model. Click **New chat** before teaching a standalone question so it can be recalled in a fresh session. [Hugging Face LoRA documentation](https://huggingface.co/docs/peft/main/package_reference/lora)

**Reset learning** removes the saved examples and adapter weights from the current profile. Use **New chat** to clear the conversation while keeping your lessons.

The desktop and terminal share `pega.sqlite3` under `%LOCALAPPDATA%\Pega` on Windows or `$XDG_DATA_HOME/pega` (default `~/.local/share/pega`) on Linux and macOS. Ordinary chat stays in memory. A saved lesson includes up to three preceding turns of context.

The terminal client remains available with `python -m pega` from the configured Python environment.

## License

Pega's source code is available under the MIT License. [License text](LICENSE)

The separately downloaded SmolLM2 base model is available under Apache 2.0. [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
