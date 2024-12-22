import { useDraftStore } from "../stores/draftStore";
import App from "../App";

export function Layout() {
  const { currentPatch } = useDraftStore();
  const lastModified = new Date().toLocaleDateString("en-US", {
    day: "numeric",
    month: "long",
  });

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground">
      <main className="flex-1">
        <App />
      </main>

      <footer className="border-t border-border/40 bg-card">
        <div className="container p-4 text-center text-sm text-muted-foreground mx-auto">
          Last model update: {lastModified} on patch {currentPatch}. Contact
          looyyd on Discord for bug reports or feature requests.
        </div>
      </footer>
    </div>
  );
}
