import { Metadata } from "next";

export const metadata: Metadata = {
  title: "How to Use Champion Recommendations in LoLDraftAI",
  description: "Learn to leverage LoLDraftAI's champion recommendation system to make better picks during draft. See real examples and get tips for maximizing this powerful feature.",
};

export default function ChampionRecommendationShowcaseLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
} 