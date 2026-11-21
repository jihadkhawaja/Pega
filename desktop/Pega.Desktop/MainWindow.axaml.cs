using System.Collections.ObjectModel;
using Avalonia.Controls;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Media;
using Avalonia.Threading;

namespace Pega.Desktop;

public sealed record ChatLine(string Content, bool IsUser)
{
    public string Speaker => IsUser ? "You" : "Pega";
}

public sealed partial class MainWindow : Window
{
    private const string KeyboardHint = "Enter to send · Shift+Enter for a new line";
    private readonly WorkerClient worker = new();
    private readonly CancellationTokenSource closing = new();
    private readonly ObservableCollection<ChatLine> messages = new();
    private int? lastReply;
    private bool learned;
    private bool emptyReply;
    private bool hasLearning;
    private bool busy;
    private bool initialized;

    public MainWindow()
    {
        InitializeComponent();
        Messages.ItemsSource = messages;
        MessageInput.AddHandler(KeyDownEvent, ComposerKeyDown, RoutingStrategies.Tunnel);
        CorrectionInput.AddHandler(KeyDownEvent, CorrectionKeyDown, RoutingStrategies.Tunnel);
        initialized = true;
        RefreshControls();
    }

    protected override async void OnOpened(EventArgs e)
    {
        base.OnOpened(e);
        await RunAsync("Starting the local model worker…", async () =>
        {
            await RequestAsync("status");
        });
        if (!closing.IsCancellationRequested)
            MessageInput.Focus();
    }

    protected override void OnClosed(EventArgs e)
    {
        closing.Cancel();
        worker.Dispose();
        closing.Dispose();
        base.OnClosed(e);
    }

    private void InputChanged(object? sender, TextChangedEventArgs e)
    {
        if (initialized)
            RefreshControls();
    }

    private async void ComposerKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.Enter && e.KeyModifiers == KeyModifiers.None)
        {
            e.Handled = true;
            await SendAsync();
        }
    }

    private void CorrectionKeyDown(object? sender, KeyEventArgs e)
    {
        if (e.Key == Key.Escape && !busy)
        {
            e.Handled = true;
            CloseCorrection();
        }
    }

    private async void SendClicked(object? sender, RoutedEventArgs e) => await SendAsync();

    private async void SuggestionClicked(object? sender, RoutedEventArgs e)
    {
        if (sender is Button { Tag: string prompt })
        {
            MessageInput.Text = prompt;
            await SendAsync();
        }
    }

    private async Task SendAsync()
    {
        string? prompt = MessageInput.Text?.Trim();
        if (busy || string.IsNullOrEmpty(prompt))
            return;

        await RunAsync("Preparing a reply…", async () =>
        {
            lastReply = null;
            learned = false;
            MessageInput.Clear();
            messages.Add(new ChatLine(prompt, IsUser: true));
            RefreshControls();
            ScrollToLatest();
            WorkerReply response;
            try
            {
                response = await RequestAsync("chat", prompt);
            }
            catch (WorkerException)
            {
                messages.RemoveAt(messages.Count - 1);
                if (string.IsNullOrWhiteSpace(MessageInput.Text))
                    MessageInput.Text = prompt;
                throw;
            }

            emptyReply = string.IsNullOrWhiteSpace(response.Answer);
            messages.Add(new ChatLine(
                emptyReply ? "Pega returned an empty reply. You can teach a replacement below." : response.Answer!,
                IsUser: false));
            lastReply = messages.Count - 1;
            ActivityText.Text = KeyboardHint;
            RefreshControls();
            ScrollToLatest();
        });
        if (!closing.IsCancellationRequested)
            MessageInput.Focus();
    }

    private async void NewChatClicked(object? sender, RoutedEventArgs e)
    {
        await RunAsync("Starting a fresh conversation…", async () =>
        {
            worker.AllowReconnect();
            await RequestAsync("reset");
            ClearConversation();
        });
    }

    private async void ResetLearningClicked(object? sender, RoutedEventArgs e)
    {
        if (!await new ResetLearningDialog().ShowDialog<bool>(this))
            return;

        await RunAsync("Resetting the learning profile…", async () =>
        {
            await RequestAsync("reset_learning");
            ClearConversation();
            ActivityText.Text = "Learning reset";
        });
    }

    private void ClearConversation()
    {
        messages.Clear();
        lastReply = null;
        learned = false;
        MessageInput.Clear();
        CloseCorrection();
    }

    private async void ApproveClicked(object? sender, RoutedEventArgs e)
    {
        await RunAsync("Preparing feedback…", async () =>
        {
            var reply = await RequestAsync("approve");
            learned = true;
            ActivityText.Text = reply.Changed ? "Reply learned and saved" : "This reply is already learned";
        });
    }

    private void CorrectClicked(object? sender, RoutedEventArgs e)
    {
        ErrorPanel.IsVisible = false;
        ComposerPanel.IsVisible = false;
        CorrectionPanel.IsVisible = true;
        ActivityText.Text = "Esc to cancel";
        CorrectionInput.Focus();
        RefreshControls();
    }

    private void CancelCorrectionClicked(object? sender, RoutedEventArgs e) => CloseCorrection();

    private void CloseCorrection()
    {
        CorrectionInput.Clear();
        CorrectionPanel.IsVisible = false;
        ComposerPanel.IsVisible = true;
        ActivityText.Text = KeyboardHint;
        MessageInput.Focus();
        RefreshControls();
    }

    private async void TeachClicked(object? sender, RoutedEventArgs e)
    {
        string? replacement = CorrectionInput.Text?.Trim();
        if (string.IsNullOrEmpty(replacement) || lastReply is not { } index)
            return;

        await RunAsync("Preparing your correction…", async () =>
        {
            var reply = await RequestAsync("correct", replacement);
            messages[index] = new ChatLine(replacement, IsUser: false);
            emptyReply = false;
            learned = true;
            CloseCorrection();
            ActivityText.Text = reply.Changed ? "Correction learned and saved" : "This reply is already learned";
            ScrollToLatest();
        });
    }

    private async Task RunAsync(string activity, Func<Task> run)
    {
        if (busy)
            return;
        busy = true;
        ErrorPanel.IsVisible = false;
        ActivityText.Text = activity;
        RefreshControls();
        try
        {
            await run();
            if (ActivityText.Text == activity)
                ActivityText.Text = KeyboardHint;
        }
        catch (WorkerException error) when (!closing.IsCancellationRequested)
        {
            ErrorText.Text = error.Message;
            ErrorPanel.IsVisible = true;
            if (worker.NeedsRestart || WorkerLabel.Text == "Starting")
                SetConnectionStatus("Disconnected", "#F2ABB6");
            ActivityText.Text = CorrectionPanel.IsVisible ? "Esc to cancel" : KeyboardHint;
        }
        catch (Exception error) when (closing.IsCancellationRequested &&
                                      error is OperationCanceledException or WorkerException)
        {
            // The worker is shut down when its window closes.
        }
        finally
        {
            busy = false;
            if (!closing.IsCancellationRequested)
                RefreshControls();
        }
    }

    private async Task<WorkerReply> RequestAsync(string action, string? text = null)
    {
        var reply = await worker.RequestAsync(action, text, ShowProgress, closing.Token);
        ExampleCount.Text = reply.Examples.ToString();
        UpdateCount.Text = reply.Updates.ToString();
        hasLearning = reply.Examples > 0 || reply.Updates > 0;
        SetConnectionStatus("Local session", "#D2F982");
        return reply;
    }

    private void SetConnectionStatus(string label, string color)
    {
        WorkerLabel.Text = label;
        WorkerDot.Fill = SolidColorBrush.Parse(color);
    }

    private void ShowProgress(string phase)
    {
        ActivityText.Text = phase switch
        {
            "loading" => "Loading SmolLM2. The first run may download the model…",
            "responding" => "Pega is writing a reply…",
            "training" => "Learning from your feedback…",
            _ => ActivityText.Text,
        };
    }

    private void RefreshControls()
    {
        Welcome.IsVisible = messages.Count == 0;
        Transcript.IsVisible = messages.Count > 0;
        FeedbackBar.IsVisible = lastReply is not null;
        NewChatButton.IsEnabled = !busy;
        ResetLearningButton.IsEnabled = !busy && hasLearning && !worker.NeedsRestart;
        SendButton.IsEnabled = !busy && !string.IsNullOrWhiteSpace(MessageInput.Text);
        ExplainButton.IsEnabled = !busy;
        PlanButton.IsEnabled = !busy;
        ApproveButton.IsEnabled = !busy && !emptyReply && !learned && !CorrectionPanel.IsVisible;
        ApproveLabel.Text = learned ? "Learned" : "Approve";
        CorrectButton.IsEnabled = !busy && !CorrectionPanel.IsVisible;
        TeachButton.IsEnabled = !busy && !string.IsNullOrWhiteSpace(CorrectionInput.Text);
        CorrectionInput.IsReadOnly = busy;
        CancelCorrectionButton.IsEnabled = !busy;
    }

    private void ScrollToLatest() => Dispatcher.UIThread.Post(() => Transcript.ScrollToEnd(), DispatcherPriority.Loaded);
}
