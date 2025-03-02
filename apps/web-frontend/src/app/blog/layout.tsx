import { Metadata } from "next";

export const metadata: Metadata = {
  title: "LoLDraftAI Blog | League Drafting Insights",
  description: "Read the latest articles, guides, and insights about League of Legends drafting strategies and how to leverage LoLDraftAI to improve your gameplay.",
};

export default function BlogLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
} 