"use client";

import Link from "next/link";
import { Visualizer } from "../components/LandingPageVisualizer";
import {
  StarIcon,
  SparklesIcon,
  LightBulbIcon,
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
            Draftking
          </h1>
          <p className="max-w-2xl text-xl text-center text-muted-foreground">
            The most accurate League of Legends draft tool, powered by AI{" "}
            <SparklesIcon className="inline-block h-5 w-5 mb-1" />
          </p>
        </div>
      </div>

      {/* Main Content */}
      <div className="container flex flex-col items-center justify-center px-4 py-12 space-y-24">
        {/* Why Draftking Section */}
        <section className="flex flex-col items-center max-w-4xl">
          <h2 className="text-3xl font-bold mb-8">
            Why is <span className="brand-text">Draftking</span> the most
            accurate draft tool?
          </h2>
          <div className="grid gap-8 md:grid-cols-2 items-center">
            <div>
              <p className="text-lg">
                <span className="brand-text">Draftking</span> is the best draft
                tool because it doesn&apos;t rely on champion statistics to
                predict the game.
              </p>
              <p className="text-lg mt-8">
                Instead, the <span className="brand-text">Draftking</span> model
                learns the full complexity of League of Legends game dynamics.
                This enables the model to make predictions not just based on
                lane matchups, but matchups against the entire ennemy team as
                well as ally champion synergies and anti-synergies, team damage
                distributions, late vs early game dynamics, and so on.
              </p>
              <div className="mt-12 w-full flex justify-center">
                <AnimatedButton href="/draft">
                  Analyse a Draft Now
                </AnimatedButton>
              </div>
            </div>
            <Visualizer />
          </div>
        </section>

        {/* Champion Recommendations Section */}
        <section className="w-full bg-secondary/10 rounded-2xl p-8">
          <div className="flex flex-col items-center max-w-4xl mx-auto">
            <h2 className="text-3xl font-bold mb-6">
              Champion recommendations{" "}
              <LightBulbIcon className="inline-block h-7 w-7 mb-2" />
            </h2>
            <p className="text-lg text-center">
              <span className="brand-text">Draftking </span> can help you pick
              the best champion for your game!{" "}
              <span className="brand-text">Drafking </span>
              enables you to add champions to favorite{" "}
              <StarIcon
                className="inline-block h-5 w-5 text-yellow-500"
                stroke="black"
                strokeWidth={2}
              />{" "}
              for a position. You can then ask{" "}
              <span className="brand-text">Draftking</span> to recommend you the
              best champion for your game!
            </p>
          </div>
        </section>

        {/* Desktop Version Section */}
        <section className="flex flex-col items-center max-w-4xl">
          <h2 className="text-3xl font-bold mb-6">
            Desktop version{" "}
            <FaWindows className="inline-block h-7 w-7 mb-1 ml-1" />
          </h2>
          <p className="text-lg text-center">
            <span className="brand-text">Draftking</span> is also available as a
            Windows desktop application. The desktop application can connect
            with the League of Legends client to access{" "}
            <span className="inline-flex items-center">
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
      </div>
    </main>
  );
}
