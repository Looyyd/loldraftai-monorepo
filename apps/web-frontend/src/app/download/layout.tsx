import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Download LoLDraftAI Desktop App",
  description: "Download the official LoLDraftAI desktop application for Windows. Get real-time draft analysis with live game tracking directly from your League client.",
};

export default function DownloadLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
} 