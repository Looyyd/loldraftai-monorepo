"use client";

import React, { forwardRef, useRef, useEffect, useState } from "react";
import { cn } from "@draftking/ui/lib/utils";
import { AnimatedBeam } from "@draftking/ui/components/ui/animated-beam";
import { champions } from "@draftking/ui/lib/champions";

const Circle = forwardRef<
  HTMLDivElement,
  { className?: string; children?: React.ReactNode }
>(({ className, children }, ref) => {
  return (
    <div
      ref={ref}
      className={cn(
        "z-10 flex size-12 items-center justify-center text-center rounded-full border-2 border-secondary bg-background p-3 text-foreground shadow-[0_0_20px_-12px_rgba(255,255,255,0.2)]",
        className
      )}
    >
      {children}
    </div>
  );
});

Circle.displayName = "Circle";

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

  // State for champion names and win rate
  const [leftChampions, setLeftChampions] = useState<string[]>([]);
  const [rightChampions, setRightChampions] = useState<string[]>([]);
  const [winRate, setWinRate] = useState(50);

  // Function to get random champions
  const getRandomChampions = () => {
    const shuffled = [...champions].sort(() => Math.random() - 0.5);
    return {
      left: shuffled.slice(0, 5).map((c) => c.name),
      right: shuffled.slice(5, 10).map((c) => c.name),
    };
  };

  // Function to get random win rate between 30-70
  const getRandomWinRate = () => Math.floor(Math.random() * (70 - 30 + 1)) + 30;

  // Effect for animation
  useEffect(() => {
    const interval = setInterval(() => {
      const { left, right } = getRandomChampions();
      setLeftChampions(left);
      setRightChampions(right);
      setWinRate(getRandomWinRate());
    }, 500); // Update every 1.5 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      className="relative flex h-[500px] w-full text-xs items-center justify-center overflow-hidden rounded-lg border bg-background p-10"
      ref={containerRef}
    >
      <div className="flex size-full items-center justify-between">
        {/* Left side champions */}
        <div className="flex flex-col gap-4">
          {[left1Ref, left2Ref, left3Ref, left4Ref, left5Ref].map((ref, i) => (
            <Circle key={i} ref={ref}>
              <span className="text-xs">{leftChampions[i] || "..."}</span>
            </Circle>
          ))}
        </div>

        {/* Center win rate */}
        <Circle
          ref={centerRef}
          className="size-16 bg-background border-[hsl(var(--chart-1))] text-[hsl(var(--chart-1))] font-semibold text-lg"
        >
          {winRate}%
        </Circle>

        {/* Right side champions */}
        <div className="flex flex-col gap-4">
          {[right1Ref, right2Ref, right3Ref, right4Ref, right5Ref].map(
            (ref, i) => (
              <Circle key={i} ref={ref}>
                <span className="text-xs">{rightChampions[i] || "..."}</span>
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
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={left2Ref}
        toRef={centerRef}
        curvature={-15}
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={left3Ref}
        toRef={centerRef}
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={left4Ref}
        toRef={centerRef}
        curvature={15}
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={left5Ref}
        toRef={centerRef}
        curvature={30}
      />

      {/* Right side beams */}
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={right1Ref}
        toRef={centerRef}
        curvature={30}
        reverse
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={right2Ref}
        toRef={centerRef}
        curvature={15}
        reverse
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={right3Ref}
        toRef={centerRef}
        reverse
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={right4Ref}
        toRef={centerRef}
        curvature={-15}
        reverse
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={right5Ref}
        toRef={centerRef}
        curvature={-30}
        reverse
      />
    </div>
  );
}
