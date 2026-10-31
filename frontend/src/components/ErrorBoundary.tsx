import React from "react";

interface Props {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

interface State {
  hasError: boolean;
}

export class ErrorBoundary extends React.Component<Props, State> {
  state: State = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch() {
    // Swallow WebGL/render errors; fallback UI is shown.
  }

  render() {
    if (this.state.hasError) {
      return this.props.fallback ?? <div>3D view unavailable.</div>;
    }
    return this.props.children;
  }
}
