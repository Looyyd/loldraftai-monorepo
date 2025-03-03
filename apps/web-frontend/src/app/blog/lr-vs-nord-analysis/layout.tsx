import { Metadata } from "next";

export const metadata: Metadata = {
  title:
    "LR vs NORD: How LoLDraftAI Can Help to Improve Draft Preparation In Pro Play",
  description:
    "Learn how LoLDraftAI could have helped Los Ratones pick better champions in their game against NORD in the NLC 2025 winter finals with advanced draft analysis and team synergy predictions.",
};

export default function LRvsNORDAnalysisLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
