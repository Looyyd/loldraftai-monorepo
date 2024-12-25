"use client";

import { Visualizer } from "../components/LandingPageVisualizer";

export default function HomePage() {
  return (
    <main className="flex min-h-screen w-full flex-col items-center bg-background text-foreground">
      <div className="container flex flex-col items-center justify-center gap-8 px-4 py-16">
        <h1 className="text-5xl font-extrabold tracking-tight text-primary sm:text-[5rem]">
          Draftking
        </h1>
        <Visualizer />
      </div>
    </main>
  );
}
