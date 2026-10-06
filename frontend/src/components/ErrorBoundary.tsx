import { Component, type ReactNode } from "react";

interface State {
  failed: boolean;
}

/** Last-resort crash boundary so a render bug never blanks the console. */
export class ErrorBoundary extends Component<{ children: ReactNode }, State> {
  state: State = { failed: false };

  static getDerivedStateFromError(): State {
    return { failed: true };
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="mx-auto max-w-[1280px] p-8">
          <div className="rounded-[20px] border border-hairline bg-panel p-8">
            <h1 className="font-display text-2xl font-bold text-ink">
              Console fault
            </h1>
            <p className="mt-2 text-dim">
              Something broke rendering this view. Reload to reset the bench.
            </p>
            <button
              type="button"
              onClick={() => window.location.reload()}
              className="mt-4 min-h-[44px] rounded-xl bg-teal px-5 py-2.5 font-semibold text-void"
            >
              Reload console
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
