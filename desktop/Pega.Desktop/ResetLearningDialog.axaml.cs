using Avalonia.Controls;
using Avalonia.Interactivity;

namespace Pega.Desktop;

public sealed partial class ResetLearningDialog : Window
{
    public ResetLearningDialog() => InitializeComponent();

    private void CancelClicked(object? sender, RoutedEventArgs e) => Close(false);

    private void ResetClicked(object? sender, RoutedEventArgs e) => Close(true);
}
