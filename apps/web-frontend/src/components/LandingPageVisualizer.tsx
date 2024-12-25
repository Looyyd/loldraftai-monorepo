"use client";

import React, { forwardRef, useRef, useEffect, useState } from "react";
import { cn } from "@draftking/ui/lib/utils";
import { AnimatedBeam } from "@draftking/ui/components/ui/animated-beam";
import { champions } from "@draftking/ui/lib/champions";
import { motion, AnimatePresence } from "framer-motion";
import { CpuChipIcon } from "@heroicons/react/24/outline";

// TODO: still change even if it's the same name
// Animated text component
const AnimatedText = ({ text }: { text: string }) => (
  <AnimatePresence mode="wait">
    <motion.span
      key={text}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="text-xs block"
    >
      {text}
    </motion.span>
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

  // Animation timing constants
  const TEXT_TRANSITION_DURATION = 0.65; // Duration of text animation
  const BEAM_DELAY = TEXT_TRANSITION_DURATION; // Delay beams until text finishes
  const BEAM_DURATION = 3; // Duration of beam animation
  const TEXT_UPDATE_INTERVAL = 3000; // Update every 2 seconds

  // Function to get random champions
  const getRandomChampions = () => {
    const shuffled = [...champions].sort(() => Math.random() - 0.5);
    return {
      left: shuffled.slice(0, 5).map((c) => c.name),
      right: shuffled.slice(5, 10).map((c) => c.name),
    };
  };

  // Effect for animation
  useEffect(() => {
    const interval = setInterval(() => {
      const { left, right } = getRandomChampions();
      setLeftChampions(left);
      setRightChampions(right);
    }, TEXT_UPDATE_INTERVAL);

    return () => clearInterval(interval);
  }, []);

  return (
    <div
      className="relative flex h-[500px] w-full items-center justify-center overflow-hidden rounded-lg border bg-background p-10"
      ref={containerRef}
    >
      <div className="flex size-full items-center justify-between">
        {/* Left side champions */}
        <div className="flex flex-col gap-4">
          {[left1Ref, left2Ref, left3Ref, left4Ref, left5Ref].map((ref, i) => (
            <Circle key={i} ref={ref}>
              <AnimatedText text={leftChampions[i] || "..."} />
            </Circle>
          ))}
        </div>

        {/* Center icon */}
        <Circle
          ref={centerRef}
          className="size-16 bg-background border-[hsl(var(--chart-1))] border-2"
        >
          <CpuChipIcon className="size-8 text-[hsl(var(--chart-1))]" />
        </Circle>

        {/* Right side champions */}
        <div className="flex flex-col gap-4">
          {[right1Ref, right2Ref, right3Ref, right4Ref, right5Ref].map(
            (ref, i) => (
              <Circle key={i} ref={ref} className="border-secondary/50">
                <AnimatedText text={rightChampions[i] || "..."} />
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
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={left2Ref}
        toRef={centerRef}
        curvature={-15}
        delay={BEAM_DELAY}
        duration={BEAM_DURATION}
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={left3Ref}
        toRef={centerRef}
        delay={BEAM_DELAY}
        duration={BEAM_DURATION}
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={left4Ref}
        toRef={centerRef}
        curvature={15}
        delay={BEAM_DELAY}
        duration={BEAM_DURATION}
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={left5Ref}
        toRef={centerRef}
        curvature={30}
        delay={BEAM_DELAY}
        duration={BEAM_DURATION}
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
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={right2Ref}
        toRef={centerRef}
        curvature={15}
        delay={BEAM_DELAY}
        duration={BEAM_DURATION}
        reverse
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={right3Ref}
        toRef={centerRef}
        delay={BEAM_DELAY}
        duration={BEAM_DURATION}
        reverse
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={right4Ref}
        toRef={centerRef}
        curvature={-15}
        delay={BEAM_DELAY}
        duration={BEAM_DURATION}
        reverse
      />
      <AnimatedBeam
        containerRef={containerRef}
        fromRef={right5Ref}
        toRef={centerRef}
        curvature={-30}
        delay={BEAM_DELAY}
        duration={BEAM_DURATION}
        reverse
      />
    </div>
  );
}
