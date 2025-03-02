import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Draft Analysis Tool",
  description: "Analyze League of Legends drafts with our AI-powered tool. Get accurate win predictions and champion recommendations to improve your team composition.",
};

export default function DraftLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
} 