import { Metadata } from "next";

export const metadata: Metadata = {
  title: "DraftGap vs LoLDraftAI: A Detailed Comparison",
  description: "See how LoLDraftAI outperforms DraftGap. Learn why our AI model better understands complex League dynamics for superior draft analysis.",
};

export default function DraftGapComparisonLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
} 