"use client";

import Link from "next/link";
import { Visualizer } from "../components/LandingPageVisualizer";
import { StarIcon } from "@heroicons/react/24/solid";

export default function HomePage() {
  return (
    <main className="flex min-h-screen w-full flex-col items-center bg-background text-foreground">
      <div className="container flex flex-col items-center justify-center gap-8 px-4 py-16">
        <h1 className="brand-text text-5xl font-extrabold tracking-tight leading-tight text-primary sm:text-[5rem]">
          Draftking
        </h1>
        <h2 className="text-2xl font-bold">
          Why is <span className="brand-text">Draftking</span> the most accurate
          draft tool?
        </h2>
        <p className="text-lg">
          <span className="brand-text">Draftking</span> is the best draft tool
          because it doesn&apos;t rely on champion statistics to predict the
          game.
        </p>
        <Visualizer />
        <p className="text-lg">
          Instead, the <span className="brand-text">Draftking</span> model
          learns the full complexity of League of Legends game dynamics. This
          enables the model to make predictions not just based on lane matchups,
          but matchups against the entire ennemy team as well as ally champion
          synergies and anti-synergies, team damage distributions, late vs early
          game dynamics, and so on.
        </p>

        <h2 className="text-2xl font-bold">Champion recommendations !</h2>
        <p className="text-lg">
          <span className="brand-text">Draftking </span> can help you pick the
          best champion for your game!{" "}
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

        <h2 className="text-2xl font-bold">Desktop version !</h2>
        <p className="text-lg">
          <span className="brand-text">Draftking</span> is also available as a
          Windows desktop application. The desktop application can connect with
          the League of Legends client to access live game data and
          automatically track the draft for you! See the{" "}
          <Link href="/download" className="text-primary underline">
            download page
          </Link>{" "}
          for details.
        </p>
      </div>
    </main>
  );
}
