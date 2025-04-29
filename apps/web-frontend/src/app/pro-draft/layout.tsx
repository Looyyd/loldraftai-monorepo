import { Metadata } from "next";

export const metadata: Metadata = {
  title: "Pro Draft Analysis Tool",
  description:
    "Analyze League of Legends drafts with our AI-powered tool. Get accurate win predictions and champion recommendations to improve your team composition.",
  robots: {
    index: false,
    follow: false,
  },
};

export default function DraftLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <>{children}</>;
}
