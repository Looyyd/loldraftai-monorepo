"use client";

import React, { forwardRef, useRef, useEffect, useState } from "react";
import { cn } from "@draftking/ui/lib/utils";
import Image from "next/image";
import { AnimatedBeam } from "@draftking/ui/components/ui/animated-beam";
import {
  champions,
  getChampionRoles,
  sortedPatches,
} from "@draftking/ui/lib/champions";
import { motion, AnimatePresence } from "framer-motion";
import { CpuChipIcon } from "@heroicons/react/24/outline";
import type { Champion } from "@draftking/ui/lib/types";

// Animated champion icon component
const AnimatedChampionIcon = ({
  champion,
  shuffleKey,
}: {
  champion: Champion;
  shuffleKey: number;
}) => (
  <AnimatePresence mode="wait">
    <motion.div
      key={`${champion.id}-${shuffleKey}`}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="relative size-12"
    >
      <Image
        src={`/icons/champions/${champion.icon}`}
        alt={champion.name}
        fill
        className="rounded-full object-cover"
        sizes="48px"
        priority
      />
    </motion.div>
  </AnimatePresence>
);

const Circle = forwardRef<
  HTMLDivElement,
  { className?: string; children?: React.ReactNode }
>(({ className, children }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        "z-10 flex size-16 items-center justify-center rounded-full border-2 border-secondary/50 bg-background p-2 text-foreground shadow-[0_0_20px_-12px_rgba(255,255,255,0.2)]",
        className
      )}
    >
      {children}
    </div>
  );
});

Circle.displayName = "Circle";

// Create role groups once
const getRoleChampions = (patch: string = sortedPatches[0]) => {
  const roleChampions: { [key: string]: Champion[] } = {
    TOP: [],
    JUNGLE: [],
    MIDDLE: [],
    BOTTOM: [],
    UTILITY: [],
  };

  // Group champions by their primary role
  champions.forEach((champion) => {
    const roles = getChampionRoles(champion.id, patch);
    if (roles.length > 0) {
      const primaryRole = roles[0]; // Get most played role
      roleChampions[primaryRole].push(champion);
    }
  });

  return roleChampions;
};

// Cache the role groups
const roleChampions = getRoleChampions();

type ChampionsToVisualize = {
  left: Champion[];
  right: Champion[];
};

// Simplified to return only one pair of teams
const getRandomChampions = (): ChampionsToVisualize => {
  // For subsequent renders, shuffle and use role-based selection
  const shuffledRoles = Object.fromEntries(
    Object.entries(roleChampions).map(([role, champs]) => [
      role,
      [...champs].sort(() => Math.random() - 0.5),
    ])
  );

  return {
    left: [
      shuffledRoles.TOP[0],
      shuffledRoles.JUNGLE[0],
      shuffledRoles.MIDDLE[0],
      shuffledRoles.BOTTOM[0],
      shuffledRoles.UTILITY[0],
    ],
    right: [
      shuffledRoles.TOP[1],
      shuffledRoles.JUNGLE[1],
      shuffledRoles.MIDDLE[1],
      shuffledRoles.BOTTOM[1],
      shuffledRoles.UTILITY[1],
    ],
  };
};

// Simple prefetcher component
const ImagePrefetcher = ({
  champions,
}: {
  champions: { left: Champion[]; right: Champion[] };
}) => {
  return (
    <div className="hidden">
      {[...champions.left, ...champions.right].map((champion) => (
        <Image
          key={champion.id}
          src={`/icons/champions/${champion.icon}`}
          alt={champion.name}
          width={48}
          height={48}
          priority
        />
      ))}
    </div>
  );
};

export function Visualizer() {
  const containerRef = useRef<HTMLDivElement>(null);
  // Left side champions
  const left1Ref = useRef<HTMLDivElement>(null);
  const left2Ref = useRef<HTMLDivElement>(null);
  const left3Ref = useRef<HTMLDivElement>(null);
  const left4Ref = useRef<HTMLDivElement>(null);
  const left5Ref = useRef<HTMLDivElement>(null);

  // Right side champions
  const right1Ref = useRef<HTMLDivElement>(null);
  const right2Ref = useRef<HTMLDivElement>(null);
  const right3Ref = useRef<HTMLDivElement>(null);
  const right4Ref = useRef<HTMLDivElement>(null);
  const right5Ref = useRef<HTMLDivElement>(null);

  // Center win rate
  const centerRef = useRef<HTMLDivElement>(null);

  // First champions should be deterministic to help SSR and hydration
  const [currentChampions, setCurrentChampions] =
    useState<ChampionsToVisualize>(() => ({
      left: [
        roleChampions.TOP[0],
        roleChampions.JUNGLE[0],
        roleChampions.MIDDLE[0],
        roleChampions.BOTTOM[0],
        roleChampions.UTILITY[0],
      ],
      right: [
        roleChampions.TOP[1],
        roleChampions.JUNGLE[1],
        roleChampions.MIDDLE[1],
        roleChampions.BOTTOM[1],
        roleChampions.UTILITY[1],
      ],
    }));

  const [nextChampions, setNextChampions] = useState<ChampionsToVisualize>(
    () => ({
      left: [
        roleChampions.TOP[2],
        roleChampions.JUNGLE[2],
        roleChampions.MIDDLE[2],
        roleChampions.BOTTOM[2],
        roleChampions.UTILITY[2],
      ],
      right: [
        roleChampions.TOP[3],
        roleChampions.JUNGLE[3],
        roleChampions.MIDDLE[3],
        roleChampions.BOTTOM[3],
        roleChampions.UTILITY[3],
      ],
    })
  );

  // Animation timing constants
  const BEAM_DELAY = 0.65 as const; // Delay beams until text finishes
  const BEAM_DURATION = 1.5 as const; // Duration of beam animation
  const TEXT_UPDATE_INTERVAL = 1500 as const; // Update every 3 seconds

  // Add a shuffle counter
  const [shuffleCount, setShuffleCount] = useState(0);

  // Effect for animation
  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentChampions(nextChampions);
      const newNext = getRandomChampions();
      setNextChampions(newNext);
      setShuffleCount((prev) => prev + 1);
    }, TEXT_UPDATE_INTERVAL);

    return () => clearInterval(interval);
  }, [nextChampions]);

  return (
    <>
      <div
        className="relative flex h-[500px] w-full max-w-screen-md items-center justify-center overflow-hidden rounded-lg border bg-background p-10"
        ref={containerRef}
      >
        <div className="flex size-full items-center justify-between">
          {/* Left side champions */}
          <div className="flex flex-col gap-4">
            {[left1Ref, left2Ref, left3Ref, left4Ref, left5Ref].map(
              (ref, i) => (
                <Circle key={i} ref={ref}>
                  {currentChampions.left[i] && (
                    <AnimatedChampionIcon
                      champion={currentChampions.left[i]}
                      shuffleKey={shuffleCount}
                    />
                  )}
                </Circle>
              )
            )}
          </div>

          {/* Center icon */}
          <Circle
            ref={centerRef}
            className="size-20 bg-background border-[hsl(var(--chart-1))] border-2"
          >
            <CpuChipIcon className="size-10 text-[hsl(var(--chart-1))]" />
          </Circle>

          {/* Right side champions */}
          <div className="flex flex-col gap-4">
            {[right1Ref, right2Ref, right3Ref, right4Ref, right5Ref].map(
              (ref, i) => (
                <Circle key={i} ref={ref}>
                  {currentChampions.right[i] && (
                    <AnimatedChampionIcon
                      champion={currentChampions.right[i]}
                      shuffleKey={shuffleCount}
                    />
                  )}
                </Circle>
              )
            )}
          </div>
        </div>

        {/* Left side beams */}
        <AnimatedBeam
          containerRef={containerRef}
          fromRef={left1Ref}
          toRef={centerRef}
          curvature={-30}
          delay={BEAM_DELAY}
          duration={BEAM_DURATION}
          startXOffset={8}
          endXOffset={-8}
        />
        <AnimatedBeam
          containerRef={containerRef}
          fromRef={left2Ref}
          toRef={centerRef}
          curvature={-15}
          delay={BEAM_DELAY}
          duration={BEAM_DURATION}
          startXOffset={8}
          endXOffset={-8}
        />
        <AnimatedBeam
          containerRef={containerRef}
          fromRef={left3Ref}
          toRef={centerRef}
          delay={BEAM_DELAY}
          duration={BEAM_DURATION}
          startXOffset={8}
          endXOffset={-8}
        />
        <AnimatedBeam
          containerRef={containerRef}
          fromRef={left4Ref}
          toRef={centerRef}
          curvature={15}
          delay={BEAM_DELAY}
          duration={BEAM_DURATION}
          startXOffset={8}
          endXOffset={-8}
        />
        <AnimatedBeam
          containerRef={containerRef}
          fromRef={left5Ref}
          toRef={centerRef}
          curvature={30}
          delay={BEAM_DELAY}
          duration={BEAM_DURATION}
          startXOffset={8}
          endXOffset={-8}
        />

        {/* Right side beams */}
        <AnimatedBeam
          containerRef={containerRef}
          fromRef={right1Ref}
          toRef={centerRef}
          curvature={30}
          delay={BEAM_DELAY}
          duration={BEAM_DURATION}
          reverse
          startXOffset={8}
          endXOffset={-8}
        />
        <AnimatedBeam
          containerRef={containerRef}
          fromRef={right2Ref}
          toRef={centerRef}
          curvature={15}
          delay={BEAM_DELAY}
          duration={BEAM_DURATION}
          reverse
          startXOffset={8}
          endXOffset={-8}
        />
        <AnimatedBeam
          containerRef={containerRef}
          fromRef={right3Ref}
          toRef={centerRef}
          delay={BEAM_DELAY}
          duration={BEAM_DURATION}
          reverse
          startXOffset={8}
          endXOffset={-8}
        />
        <AnimatedBeam
          containerRef={containerRef}
          fromRef={right4Ref}
          toRef={centerRef}
          curvature={-15}
          delay={BEAM_DELAY}
          duration={BEAM_DURATION}
          reverse
          startXOffset={8}
          endXOffset={-8}
        />
        <AnimatedBeam
          containerRef={containerRef}
          fromRef={right5Ref}
          toRef={centerRef}
          curvature={-30}
          delay={BEAM_DELAY}
          duration={BEAM_DURATION}
          reverse
          startXOffset={8}
          endXOffset={-8}
        />
      </div>

      {/* Simple prefetcher for next batch */}
      <ImagePrefetcher champions={nextChampions} />
    </>
  );
}
