using System.ComponentModel;
using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace Pega.Desktop;

internal sealed record WorkerReply(
    string Kind, string? Phase, string? Answer, string? Message,
    int Examples, int Updates, bool Changed);

internal sealed class WorkerException(string message, Exception? inner = null) : Exception(message, inner);

internal sealed class WorkerClient : IDisposable
{
    private static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.Web);
    private static readonly Encoding WireEncoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);
    private Process? process;
    private string? lastDiagnostic;

    public bool NeedsRestart { get; private set; }

    public async Task<WorkerReply> RequestAsync(
        string action, string? text, Action<string> onProgress, CancellationToken cancellationToken)
    {
        if (NeedsRestart)
            throw new WorkerException("Start a new chat to reconnect to the local model.");

        var worker = process ??= Start();
        try
        {
            var request = JsonSerializer.Serialize(new { action, text }, JsonOptions);
            await worker.StandardInput.WriteLineAsync(request.AsMemory(), cancellationToken);
            await worker.StandardInput.FlushAsync(cancellationToken);
            while (await worker.StandardOutput.ReadLineAsync(cancellationToken) is { } line)
            {
                var reply = JsonSerializer.Deserialize<WorkerReply>(line, JsonOptions);
                switch (reply?.Kind)
                {
                    case "progress" when reply.Phase is { } phase:
                        onProgress(phase);
                        break;
                    case "result":
                        return reply;
                    case "error":
                        throw new WorkerException(reply.Message ?? "The model could not complete the request.");
                    default:
                        throw new JsonException("Unexpected response from the Pega worker.");
                }
            }
            throw new IOException("The Pega worker exited before responding.");
        }
        catch (Exception error) when (error is IOException or JsonException or InvalidOperationException)
        {
            cancellationToken.ThrowIfCancellationRequested();
            NeedsRestart = true;
            StopWorker();
            string details = lastDiagnostic is { Length: > 0 } ? $" {lastDiagnostic}" : "";
            throw new WorkerException($"The local model disconnected. Start a new chat to reconnect.{details}", error);
        }
    }

    private Process Start()
    {
        var start = new ProcessStartInfo(FindPython())
        {
            WorkingDirectory = AppContext.BaseDirectory,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            StandardInputEncoding = WireEncoding,
            StandardOutputEncoding = WireEncoding,
            StandardErrorEncoding = WireEncoding,
        };
        foreach (string argument in new[] { "-u", "-m", "pega.worker" })
            start.ArgumentList.Add(argument);
        if (Environment.GetEnvironmentVariable("PEGA_OFFLINE") == "1")
            start.ArgumentList.Add("--offline");
        AddOption(start, "--data-dir", "PEGA_DATA_DIR");
        AddOption(start, "--device", "PEGA_DEVICE");
        start.Environment["PYTHONIOENCODING"] = "utf-8";
        start.Environment["PYTHONUNBUFFERED"] = "1";

        lastDiagnostic = null;
        var worker = new Process { StartInfo = start };
        worker.ErrorDataReceived += (_, args) =>
        {
            if (!string.IsNullOrWhiteSpace(args.Data))
                lastDiagnostic = args.Data[..Math.Min(args.Data.Length, 400)];
        };
        try
        {
            worker.Start();
            worker.BeginErrorReadLine();
            return worker;
        }
        catch (Win32Exception error)
        {
            worker.Dispose();
            throw new WorkerException(
                "Pega could not start Python. Set up the Python environment in the README or set PEGA_PYTHON.", error);
        }
    }

    private static void AddOption(ProcessStartInfo start, string flag, string variable)
    {
        if (Environment.GetEnvironmentVariable(variable) is { Length: > 0 } value)
        {
            start.ArgumentList.Add(flag);
            start.ArgumentList.Add(value);
        }
    }

    private static string FindPython()
    {
        if (Environment.GetEnvironmentVariable("PEGA_PYTHON") is { Length: > 0 } configured)
            return configured;

        bool windows = OperatingSystem.IsWindows();
        string executable = windows ? Path.Combine("Scripts", "python.exe") : Path.Combine("bin", "python");
        for (var directory = new DirectoryInfo(AppContext.BaseDirectory); directory is not null; directory = directory.Parent)
        {
            if (!File.Exists(Path.Combine(directory.FullName, "pega", "session.py")))
                continue;
            string candidate = Path.Combine(directory.FullName, ".venv", executable);
            if (File.Exists(candidate))
                return candidate;
        }

        if (windows)
        {
            string localData = Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData);
            string previousRuntime = Path.Combine(localData, "PegaRuntimeCUDA", executable);
            if (File.Exists(previousRuntime))
                return previousRuntime;
        }
        return windows ? "python" : "python3";
    }

    public void AllowReconnect() => NeedsRestart = false;

    private void StopWorker()
    {
        if (process is null)
            return;
        try
        {
            if (!process.HasExited)
                process.Kill(entireProcessTree: true);
        }
        catch (Exception error) when (error is InvalidOperationException or Win32Exception)
        {
            // The worker may exit before the termination request.
        }
        finally
        {
            process.Dispose();
            process = null;
        }
    }

    public void Dispose() => StopWorker();
}
