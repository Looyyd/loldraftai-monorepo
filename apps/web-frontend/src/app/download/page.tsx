"use client";

import React from "react";
import { Button } from "@draftking/ui/components/ui/button";
import { track } from "@vercel/analytics";
import { Download, AlertTriangle } from "lucide-react";
import { FaWindows } from "react-icons/fa";

export default function DownloadPage() {
  return (
    <main className="flex min-h-[calc(100vh-200px)] flex-col items-center justify-center p-4 sm:p-8">
      <div className="max-w-2xl text-center space-y-6 sm:space-y-8">
        <h1 className="text-2xl sm:text-4xl font-bold flex items-center justify-center gap-2">
          Download <span className="brand-text">LoLDraftAI</span>{" "}
          <FaWindows className="h-5 w-5 sm:h-7 sm:w-7" />
        </h1>
        <p className="text-base sm:text-lg text-muted-foreground">
          Download the latest version of{" "}
          <span className="brand-text">LoLDraftAI</span> for Windows to analyze
          your League of Legends games with{" "}
          <span className="inline-flex font-bold items-center">
            live{" "}
            <span className="relative flex h-2 w-2 ml-1">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500"></span>
            </span>
          </span>{" "}
          game tracking.
        </p>
        <div className="bg-yellow-200 text-yellow-950 p-3 sm:p-4 rounded-md border border-yellow-300 space-y-2">
          <h2 className="font-semibold text-base sm:text-lg flex items-center gap-2">
            <AlertTriangle className="h-4 w-4 sm:h-5 sm:w-5" />
            Before You Install
          </h2>
          <p className="text-xs sm:text-sm">
            Since <span className="brand-text">LoLDraftAI</span> is currently in
            beta, Windows SmartScreen might display a warning during
            installation. This is normal because:
          </p>
          <ul className="text-xs sm:text-sm list-disc list-inside space-y-1">
            <li>
              We&apos;re a new application that hasn&apos;t built up a
              reputation with Microsoft yet
            </li>
            <li>
              The app only needs access to read your League client data -
              nothing else
            </li>
          </ul>
          <p className="text-xs sm:text-sm mt-2">
            To proceed with installation, click &quot;More info&quot; →
            &quot;Run anyway&quot; when prompted.
          </p>
        </div>
        <Button
          asChild
          size="lg"
          className="gap-2 text-sm sm:text-base w-full sm:w-auto"
          onClick={() => track("Download Request")}
        >
          <a
            href="https://releases.loldraftai.com/latest/LoLDraftAI.Setup.exe"
            download
          >
            <Download className="h-5 w-5" />
            Download for Windows
          </a>
        </Button>
      </div>
    </main>
  );
}
