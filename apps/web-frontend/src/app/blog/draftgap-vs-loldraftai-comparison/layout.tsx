import { Metadata } from "next";

export const metadata: Metadata = {
  title: "DraftGap vs LoLDraftAI: A Detailed Comparison",
  description: "See how LoLDraftAI outperforms DraftGap with 65.6% prediction accuracy vs 56.5%. Learn why our AI model better understands complex League dynamics for superior draft analysis.",
};

export default function DraftGapComparisonLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
} 