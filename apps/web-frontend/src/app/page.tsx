"use client";

import Link from "next/link";
import { Visualizer } from "../components/LandingPageVisualizer";
import { FeaturesShowcase } from "../components/FeaturesShowcase";
import {
  StarIcon,
  SparklesIcon,
  LightBulbIcon,
  TrophyIcon,
} from "@heroicons/react/24/solid";
import { FaWindows } from "react-icons/fa";
import { AnimatedButton } from "../components/AnimatedButton";

export default function HomePage() {
  return (
    <main className="flex min-h-screen w-full flex-col items-center bg-background text-foreground">
      {/* Hero Section */}
      <div className="w-full bg-gradient-to-b from-primary/10 to-background py-8">
        <div className="container flex flex-col items-center justify-center gap-6 px-4">
          <h1 className="brand-text text-5xl font-extrabold tracking-tight leading-tight text-primary text-center sm:text-[5rem]">
            LoLDraftAI
          </h1>
          <p className="max-w-2xl text-xl text-center text-muted-foreground">
            The most accurate League of Legends draft tool, powered by AI{" "}
            <SparklesIcon className="inline-block h-5 w-5 mb-1" />
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="container flex flex-col items-center justify-center px-4 py-12 space-y-12 lg:space-y-24">
        {/* Why LoLDraftAI Section */}
        <section className="flex flex-col items-center max-w-4xl">
          <h2 className="text-3xl font-bold mb-8 text-center">
            Why is <span className="brand-text">LoLDraftAI</span> the most
            accurate draft tool?
          </h2>
          <div className="grid gap-8 md:grid-cols-2 items-center">
            <div>
              <p className="text-lg">
                <span className="brand-text">LoLDraftAI</span> is the best
                League of Legends draft tool because it doesn&apos;t rely on
                champion statistics to predict the game.
              </p>
              <p className="text-lg mt-8">
                Instead, the <span className="brand-text">LoLDraftAI</span>{" "}
                model learns and{" "}
                <span className="font-bold">
                  makes predictions based on the full complexity of League of
                  Legends game dynamics,
                </span>{" "}
                such as in context matchups, ally champion synergies and
                anti-synergies, team damage distributions, late vs early game
                dynamics, etc.
              </p>
              <p className="text-lg mt-4">
                <Link
                  href="/blog/draftgap-vs-loldraftai-comparison"
                  className="text-primary hover:underline"
                >
                  See how our draft AI outperforms other tools like DraftGap in
                  our detailed comparison.
                </Link>
              </p>
              <div className="mt-12 w-full flex justify-center">
                <AnimatedButton href="/draft">
                  Analyse a Draft Now
                </AnimatedButton>
              </div>
            </div>
            <div className="flex flex-col items-center gap-2">
              <Visualizer />
              <p className="text-sm text-muted-foreground italic text-center">
                LoLDraftAI understands the full complexity of champions, not
                just statistics!
              </p>
            </div>
          </div>
        </section>

        {/* Champion Recommendations Section */}
        <section className="w-full bg-secondary/10 rounded-2xl">
          <div className="flex flex-col items-center max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold text-center mb-6">
              Champion{" "}
              <span className="whitespace-nowrap">
                recommendations
                <LightBulbIcon className="inline-block h-7 w-7 mb-2 ml-1" />
              </span>
            </h2>
            <p className="text-lg text-jusify">
              <span className="brand-text">LoLDraftAI </span>{" "}
              <span className="font-bold">
                {" "}
                can help you pick the best champion for your game!{" "}
              </span>
              <span className="brand-text">LoLDraftAI </span>
              enables you to add champions to your{" "}
              <span className="inline-flex font-bold items-center">
                favorites{" "}
                <StarIcon
                  className="inline-block h-5 w-5 text-yellow-500"
                  stroke="black"
                  strokeWidth={2}
                />{" "}
              </span>{" "}
              for a position. You can then ask{" "}
              <span className="brand-text">LoLDraftAI</span> to recommend you
              the best champion for your game!
            </p>
          </div>
        </section>

        {/* Desktop Version Section */}
        <section className="flex flex-col items-center max-w-4xl">
          <h2 className="text-3xl font-bold text-center mb-6">
            Desktop{" "}
            <span className="whitespace-nowrap">
              version
              <FaWindows className="inline-block h-7 w-7 mb-1 ml-1" />
            </span>
          </h2>
          <p className="text-lg text-justify">
            <span className="brand-text">LoLDraftAI</span>{" "}
            <span className="font-bold">
              {" "}
              is also available as a Windows desktop application.{" "}
            </span>
            The desktop application can connect with the League of Legends
            client to access{" "}
            <span className="inline-flex font-bold items-center">
              live{" "}
              <span className="relative flex h-2 w-2 ml-1">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex h-2 w-2 rounded-full bg-red-500"></span>
              </span>
            </span>{" "}
            game data and automatically track the draft for you! See the{" "}
            <Link href="/download" className="text-primary underline">
              download page
            </Link>{" "}
            for details.
          </p>
        </section>

        {/* How to win your games using LoLDraftAI Section */}
        <section className="w-full">
          <h2 className="text-3xl font-bold text-center mb-2">
            How to win your games using{" "}
            <span className="brand-text">LoLDraftAI</span>{" "}
            <span className="whitespace-nowrap">
              <TrophyIcon className="inline-block h-7 w-7 mb-1 ml-1" />
            </span>
          </h2>
          <FeaturesShowcase />
        </section>

        {/* Added CTA Section */}
        <section className="flex flex-col items-center">
          <AnimatedButton href="/draft">Analyse a Draft Now</AnimatedButton>
        </section>
      </div>
    </main>
  );
}
