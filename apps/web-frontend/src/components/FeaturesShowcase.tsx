"use client";

import * as React from "react";
import CloudFlareImage from "@/components/CloudFlareImage";
import {
  Carousel,
  CarouselContent,
  CarouselItem,
  CarouselNext,
  CarouselPrevious,
  type CarouselApi,
} from "@draftking/ui/components/ui/carousel";

// Define the content for each slide
const SHOWCASE_ITEMS = [
  {
    image: "/showcase/image1.png",
    title: "Use live tracking to follow your draft",
    description:
      "The LoLDraftAI desktop application can connect with the League of Legends client to access live game data and automatically track the draft for you!",
  },
  {
    image: "/showcase/image2.png",
    title: "Get champion recommendations",
    description:
      "Let LoLDraftAI recommend the best champions for you and start your game with a head start!",
  },
  {
    image: "/showcase/image3.png",
    title: "Judge the final draft",
    description:
      'Understand the strengths and weaknesses of your team composition. "gg draft gap"!',
  },
];

export function FeaturesShowcase() {
  const [api, setApi] = React.useState<CarouselApi>();
  const [current, setCurrent] = React.useState(0);
  const [count, setCount] = React.useState(0);

  React.useEffect(() => {
    if (!api) return;

    setCount(api.scrollSnapList().length);
    setCurrent(api.selectedScrollSnap() + 1);

    api.on("select", () => {
      setCurrent(api.selectedScrollSnap() + 1);
    });
  }, [api]);

  return (
    <div className="w-full max-w-4xl mx-auto">
      <Carousel setApi={setApi} className="w-full">
        <CarouselContent>
          {SHOWCASE_ITEMS.map((item, index) => (
            <CarouselItem key={index}>
              <div className="p-1">
                <div className="rounded-xl overflow-hidden flex flex-col h-full">
                  <div className="relative aspect-video flex items-center justify-center bg-black/5">
                    <CloudFlareImage
                      src={item.image}
                      alt={item.title}
                      width={1920}
                      height={1080}
                      className="w-full h-full object-contain"
                    />
                  </div>
                  <div className="p-6 bg-card">
                    <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                    <p className="text-muted-foreground">{item.description}</p>
                  </div>
                </div>
              </div>
            </CarouselItem>
          ))}
        </CarouselContent>
        <CarouselPrevious className="left-4" />
        <CarouselNext className="right-4" />
      </Carousel>
      <div className="py-2 text-center text-sm text-muted-foreground">
        {current} / {count}
      </div>
    </div>
  );
}
