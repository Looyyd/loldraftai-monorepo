import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Correction: Reddit Post Accuracy Claims",
  description: "A correction regarding our previously claimed 62% accuracy statistics that were affected by overfitting issues.",
};

export default function CorrectionRedditPostLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
