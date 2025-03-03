import { useDraftStore } from "../stores/draftStore";
import App from "../App";
import { Toaster } from "@draftking/ui/components/ui/toaster";
import { useEffect, useState } from "react";
import { VERCEL_URL } from "../utils";

export function Layout() {
  const { currentPatch } = useDraftStore();
  const [lastModified, setLastModified] = useState<string>("");

  useEffect(() => {
    const fetchMetadata = async () => {
      try {
        const response = await fetch(`${VERCEL_URL}/api/metadata`);
        if (!response.ok) {
          throw new Error("Failed to fetch metadata");
        }
        const data = await response.json();
        const date = new Date(data.last_modified);

        if (isNaN(date.getTime())) {
          throw new Error("Invalid date received from server");
        }

        const formattedDate = date.toLocaleDateString("en-US", {
          day: "numeric",
          month: "long",
        });
        setLastModified(formattedDate);
      } catch (error) {
        console.error("Error fetching metadata:", error);
        // Fallback to current date if fetch fails
        setLastModified(
          new Date().toLocaleDateString("en-US", {
            day: "numeric",
            month: "long",
          })
        );
      }
    };

    fetchMetadata();
  }, []); // Run once on component mount

  return (
    <div className="flex min-h-screen flex-col bg-background text-foreground font-sans">
      <main className="flex-1">
        <App />
      </main>

      <footer className="border-t border-border/40 bg-neutral-950">
        <div className="container p-4 text-center text-sm text-muted-foreground mx-auto">
          <div className="mb-2">
            Last model update: {lastModified} on patch {currentPatch}. After a
            new patch, expect a few days of delay before an update.{" "}
            <a
              href="https://discord.gg/MpbtNEwTT7"
              className="text-blue-400 hover:underline"
              target="_blank"
              rel="noopener noreferrer"
            >
              Join our Discord
            </a>{" "}
            for bug reports or feature requests.
          </div>
          <div className="text-xs">
            LoLDraftAI isn't endorsed by Riot Games and doesn't reflect the
            views or opinions of Riot Games or anyone officially involved in
            producing or managing Riot Games properties. Riot Games, and all
            associated properties are trademarks or registered trademarks of
            Riot Games, Inc.
          </div>
        </div>
      </footer>
      <Toaster />
    </div>
  );
}
