"use client";

import Link from "next/link";
import { CalendarIcon } from "@heroicons/react/24/solid";

interface BlogPost {
  title: string;
  description: string;
  date: string;
  slug: string;
}

// Helper component to brand the text LoLDraftAI
const BrandedText = ({ text }: { text: string }) => {
  return text.split("LoLDraftAI").map((part, i, arr) => (
    <span key={i}>
      {part}
      {i < arr.length - 1 && <span className="brand-text">LoLDraftAI</span>}
    </span>
  ));
};

const blogPosts: BlogPost[] = [
  {
    title:
      "LR vs NORD: How LoLDraftAI Can Help to Improve Draft Preparation In Pro Play",
    description:
      "Analysis of how LoLDraftAI could have helped LosRatones make better champion selections in their NLC 2025 finals match against NORD, focusing on the suboptimal Rakan pick.",
    date: "2025-03-03",
    slug: "lr-vs-nord-analysis",
  },
  {
    title: "DraftGap vs LoLDraftAI: A Detailed Comparison",
    description:
      "Through statistical validation, we've found that LoLDraftAI consistently outperforms DraftGap in prediction accuracy (65.6% vs 56.5% on unseen data).",
    date: "2025-03-02",
    slug: "draftgap-vs-loldraftai-comparison",
  },
  {
    title: "How to Use Champion Recommendations in LoLDraftAI",
    description:
      "Learn how to leverage LoLDraftAI's powerful champion recommendation system to improve your drafting strategy and win more games.",
    date: "2025-02-16",
    slug: "champion-recommendation-showcase",
  },
];

export default function BlogPage() {
  return (
    <main className="flex min-h-screen w-full flex-col items-center bg-background text-foreground">
      {/* Header Section */}
      <div className="w-full bg-gradient-to-b from-primary/10 to-background py-8">
        <div className="container flex flex-col items-center justify-center gap-4 px-4">
          <h1 className="text-4xl font-bold tracking-tight text-primary text-center">
            <span className="brand-text">LoLDraftAI</span> Blog
          </h1>
          <p className="text-xl text-center text-muted-foreground">
            Insights and guides to help you master League drafting
          </p>
        </div>
      </div>

      {/* Blog Posts List */}
      <div className="container px-4 py-12">
        <div className="max-w-3xl mx-auto space-y-8">
          {blogPosts.map((post) => (
            <Link
              key={post.slug}
              href={`/blog/${post.slug}`}
              className="block group"
            >
              <article className="p-6 rounded-lg border border-border hover:border-primary transition-colors duration-200">
                <h2 className="text-2xl font-bold group-hover:text-primary transition-colors duration-200">
                  <BrandedText text={post.title} />
                </h2>
                <div className="flex items-center gap-2 mt-2 text-sm text-muted-foreground">
                  <CalendarIcon className="h-4 w-4" />
                  <time dateTime={post.date}>
                    {new Date(post.date).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                      day: "numeric",
                    })}
                  </time>
                </div>
                <p className="mt-4 text-muted-foreground">
                  <BrandedText text={post.description} />
                </p>
              </article>
            </Link>
          ))}
        </div>
      </div>
    </main>
  );
}
