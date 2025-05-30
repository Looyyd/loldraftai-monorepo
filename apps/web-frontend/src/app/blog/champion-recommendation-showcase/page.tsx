"use client";

import Link from "next/link";
import { ClickableImage } from "@/components/ClickableImage";
import BlogLayout from "@/components/BlogLayout";

export default function ChampionRecommendationShowcase() {
  return (
    <BlogLayout
      title="How to Use Champion Recommendations in LoLDraftAI"
      date="February 16, 2025"
    >
      <p>
        <span className="brand-text">LoLDraftAI</span>&apos;s champion
        recommendation system is a powerful tool that can help you make better
        champion selections during draft. In this post, we&apos;ll look at a
        real game that showcases how to effectively use these recommendations,
        and discuss the advantages of using our desktop application for live
        draft tracking.
      </p>
      <h2>Desktop App Advantages: Live Draft Tracking</h2>
      <p>
        While the web version is powerful, the{" "}
        <Link href="/download">desktop application</Link> offers some
        significant advantages:
      </p>
      <ul>
        <li>Automatic tracking of picks and bans in real-time</li>
        <li>Instantly updates recommendations as champions are selected</li>
        <li>Automatically greys out picked or banned champions</li>
      </ul>
      <h2>Game Showcase: Perfect Taric Counter-Pick</h2>
      <p>
        In our showcase game, we&apos;ll look at a textbook example of using
        {` `}
        <span className="brand-text">LoLDraftAI</span> champion suggestion in a
        live game.
      </p>
      <p>
        You can see below the recording of how to use LoLDraftAI champion
        suggestion in a live game.
      </p>
      <div className="flex justify-center w-full">
        <div className="relative w-full aspect-video">
          <iframe
            className="absolute top-0 left-0 w-full h-full"
            src="https://www.youtube.com/embed/0VoN0DCACzE?si=ix3HwyauBjrEjIq4"
            title="YouTube video player"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
          ></iframe>
        </div>
      </div>
      <p>
        It&apos;s often useful to start checking suggestions early in the draft.
        This allows you to:
      </p>
      <ul>
        <li>Plan potential picks in advance</li>
        <li>
          Communicate options with your team (through prepicking in champ
          select)
        </li>
      </ul>
      <p>
        I ended up picking in R4(red fourth pick). We could already see the
        Galio/Jarvan combination at this point of the draft so I was quite
        confident the Taric pick would work out well
      </p>
      <p>
        In the final draft, Taric ended up being an excellent counter-pick for
        several reasons:
      </p>
      <ul>
        <li>Strong counter to the enemy dive composition</li>
        <li>
          Excellent synergy with our team with taric E being a good chain CC
          with Gragas, Xin Zhao, Yasuo
        </li>
        <li>
          Good lane matchup, where we can just farm and outscale the ennemy
          botlane
        </li>
      </ul>
      Here is the final draft analysis for the game. You can see Taric remains a
      great pick with 9% winrate impact (meaning without Taric, the team would
      on average have 9% less winrate):
      <ClickableImage
        src="/blog/champion-recommendation-showcase/game_1_final_analysis.png"
        alt="Draft analysis showing Taric's high impact"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      The analysis was totally correct in this game, and we won convincingly
      with Taric being a really good pick especially against their topside dive.{" "}
      <a
        href="https://www.op.gg/summoners/euw/LoLDraftAI-loyd/matches/TtNFybHTVlUkyADsLJL5GfiP8aOx0Lkez9BTPg94f7A%3D/1739121021000"
        target="_blank"
        rel="noopener noreferrer"
      >
        See game results.
      </a>
      <div className="note bg-secondary/10 p-4 rounded-lg my-6">
        <p className="text-sm">
          Note: The game was played on patch 15.03 while using a model trained
          on patch 15.02. This demonstrates an important point:
          {` `}
          <span className="brand-text">LoLDraftAI</span> remains relevant even
          when slightly behind the current patch, as meta shifts are minimal
          from patch to patch.
        </p>
      </div>
      <h2>Additional tips</h2>
      Here are two additional tips for using champion recommendations.
      <h3>Tip 1: Picking later is better</h3>
      If you have a deep champion pool, you should ask to pick later in the
      draft. The model truly shines when both team comps are mostly known,
      because it can help quickly find picks that totally counter the ennemy
      team.
      <h3>Tip 2: Pick champions that you can play well</h3>
      It may sound obvious, but it&apos;s always better to pick a champion that
      has lower winrate but that you can play well. Understand that the model is
      trained on solo queue games where most people pick champs that they are
      good at, so the predictions assume you can play the champion well.
      <div className="mt-8 p-4 bg-primary/10 rounded-lg">
        <h2 className="text-xl font-bold mb-2">Ready to Try It Yourself?</h2>
        <p>
          Experience <span className="brand-text">LoLDraftAI</span>&apos;s
          champion recommendations in your own games:
        </p>
        <ul>
          <li>
            Use the <Link href="/draft">web version</Link> for analysis
          </li>
          <li>
            Download the <Link href="/download">desktop app</Link> for live
            draft tracking
          </li>
        </ul>
      </div>
    </BlogLayout>
  );
}
