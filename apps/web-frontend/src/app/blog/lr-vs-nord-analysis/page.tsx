"use client";

import Link from "next/link";
import { ClickableImage } from "@/components/ClickableImage";
import BlogLayout from "@/components/BlogLayout";

export default function LRvsNORDAnalysis() {
  return (
    <BlogLayout
      title="LR vs NORD: How LoLDraftAI Can Help to Improve Draft Preparation In Pro Play"
      date="March 3, 2025"
    >
      <p>
        While Los Ratones(LR) showed an outstanding performance in the{" "}
        <Link
          href="https://gol.gg/game/stats/64748/page-summary/"
          target="_blank"
          className="text-blue-500 hover:underline"
        >
          NLC 2025 Winter finals
        </Link>
        , game 1 was a little bit of a struggle for Rekkless piloting Rakan. In
        this article we will show that Rakan was almost unplayable in the draft,
        what could have been a better pick and how{" "}
        <span className="brand-text">LoLDraftAI</span> can help prepare drafts
        in competitive play.
      </p>

      <h2>Why Rakan was a bad pick</h2>
      <p>
        First of all, let&apos;s see what{" "}
        <span className="brand-text">LoLDraftAI</span> thinks of the complete
        draft:
      </p>

      <ClickableImage
        src="/blog/lr-vs-nord-analysis/full-draft-analysis.png"
        alt="Full Draft Analysis by LoLDraftAI"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />

      <p>
        The overall draft is about even, with quite a bit of picks being
        suboptimal. However one thing that jumps to the eye is the -18% winrate
        impact of the Rakan. This means that on average, the same blue side
        draft with other supports would have 18% more chance of winning.
      </p>
      <p>
        Here is how I would interpret why the Rakan pick is hard to play in this
        draft:
      </p>
      <ul className="list-disc pl-6 my-4">
        <li>
          No wombo combo with Rakan: blue side doesn&apos;t have any other snap
          follow up to the Rakan engage such as an Orianna, Yasuo or Hwei.
        </li>
        <li>
          High CC in enemy team: Rakan is squishy and can be easily one shot
          when CC&apos;ed
        </li>
        <li>
          Low synergy with Ivern: Ivern is best with melee support that commit
          to the engage to get the maximum value out of Ivern E
        </li>
      </ul>

      <h2>What could have been a better pick in the moment</h2>
      <p>
        The support pick was on Blue 5th pick, with Poppy, Alistar, Braum being
        banned:
      </p>

      <ClickableImage
        src="/blog/lr-vs-nord-analysis/rakan-pick.png"
        alt="Picture of draft state during Rakan pick"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />

      <p>
        In voice comms, Rekkless was hesitating between Rell and Rakan. Here is
        the a rough transcript of the voice comms during that pick:
      </p>
      <div className="p-4 my-4 rounded">
        <p>
          <strong>Caedrel</strong>: Rell is pretty solid, Rakan?
        </p>
        <p>
          <strong>Rekkless</strong>: I&apos;m probably going to go Rakan
        </p>
        <p>
          <strong>Nemesis</strong>: Rakan is really good with Cho.
        </p>
        <p>[…]</p>
        <p>
          <strong>Rekkless</strong>: The only thing is that our support jungle
          is not very good together.
        </p>
      </div>

      <p>
        We know Rakan ended up not being a good pick, let&apos;s see what{" "}
        <span className="brand-text">LoLDraftAI</span> suggest instead.
      </p>

      <p>
        <span className="brand-text">LoLDraftAI</span> is able to generate
        champion recommendations by predicting the winrate of a draft that
        contains that champion. In the following table, I included 2 columns:
      </p>
      <ul className="list-disc pl-6 my-4">
        <li>
          <strong>Pre-Viktor WR</strong>: The predicted winrate with the
          following picks locked:
          <ul className="list-disc pl-6 my-2">
            <li>
              <strong>Blue side</strong>: Volibear, Ivern, Cho&apos;Gath, Ezreal
            </li>
            <li>
              <strong>Red side</strong>: Camille, Skarner, Ezreal, Leona
            </li>
          </ul>
        </li>
        <li>
          <strong>Post-Viktor WR</strong>: The winrate with Viktor also locked
          in for red side
        </li>
      </ul>
      <p>
        Here are the meta support suggestions, filtering out banned or picked
        champions and more &quot;solo queue&quot; picks such as Xerath or Shaco:
      </p>

      <div className="overflow-x-auto my-6">
        <table className="min-w-full border-collapse">
          <thead>
            <tr>
              <th className="border border-gray-300 px-4 py-2">Champion</th>
              <th className="border border-gray-300 px-4 py-2">
                Pre-Viktor WR
              </th>
              <th className="border border-gray-300 px-4 py-2">
                Post-Viktor WR
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Pyke</td>
              <td className="border border-gray-300 px-4 py-2">74.5%</td>
              <td className="border border-gray-300 px-4 py-2">80.0%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Thresh</td>
              <td className="border border-gray-300 px-4 py-2">68.8%</td>
              <td className="border border-gray-300 px-4 py-2">70.6%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Galio</td>
              <td className="border border-gray-300 px-4 py-2">67.7%</td>
              <td className="border border-gray-300 px-4 py-2">71.2%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Karma</td>
              <td className="border border-gray-300 px-4 py-2">66.0%</td>
              <td className="border border-gray-300 px-4 py-2">73.0%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Bard</td>
              <td className="border border-gray-300 px-4 py-2">65.9%</td>
              <td className="border border-gray-300 px-4 py-2">72.9%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Renata Glasc</td>
              <td className="border border-gray-300 px-4 py-2">64.7%</td>
              <td className="border border-gray-300 px-4 py-2">68.3%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Maokai</td>
              <td className="border border-gray-300 px-4 py-2">62.1%</td>
              <td className="border border-gray-300 px-4 py-2">64.6%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Blitzcrank</td>
              <td className="border border-gray-300 px-4 py-2">62.0%</td>
              <td className="border border-gray-300 px-4 py-2">67.2%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Pantheon</td>
              <td className="border border-gray-300 px-4 py-2">61.8%</td>
              <td className="border border-gray-300 px-4 py-2">66.3%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Tahm Kench</td>
              <td className="border border-gray-300 px-4 py-2">59.8%</td>
              <td className="border border-gray-300 px-4 py-2">68.7%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Ashe</td>
              <td className="border border-gray-300 px-4 py-2">59.2%</td>
              <td className="border border-gray-300 px-4 py-2">66.6%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Taric</td>
              <td className="border border-gray-300 px-4 py-2">58.1%</td>
              <td className="border border-gray-300 px-4 py-2">59.9%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Seraphine</td>
              <td className="border border-gray-300 px-4 py-2">57.3%</td>
              <td className="border border-gray-300 px-4 py-2">59.2%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Shen</td>
              <td className="border border-gray-300 px-4 py-2">57.3%</td>
              <td className="border border-gray-300 px-4 py-2">62.0%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Rell</td>
              <td className="border border-gray-300 px-4 py-2">55.9%</td>
              <td className="border border-gray-300 px-4 py-2">59.3%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Janna</td>
              <td className="border border-gray-300 px-4 py-2">54.0%</td>
              <td className="border border-gray-300 px-4 py-2">59.8%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Soraka</td>
              <td className="border border-gray-300 px-4 py-2">53.1%</td>
              <td className="border border-gray-300 px-4 py-2">61.6%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Lulu</td>
              <td className="border border-gray-300 px-4 py-2">52.0%</td>
              <td className="border border-gray-300 px-4 py-2">58.0%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Nautilus</td>
              <td className="border border-gray-300 px-4 py-2">50.3%</td>
              <td className="border border-gray-300 px-4 py-2">52.0%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Rakan</td>
              <td className="border border-gray-300 px-4 py-2">41.6%</td>
              <td className="border border-gray-300 px-4 py-2">47.3%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Yuumi</td>
              <td className="border border-gray-300 px-4 py-2">31.2%</td>
              <td className="border border-gray-300 px-4 py-2">34.8%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        It is quite telling that Rakan is basically the worst pick besides
        Yuumi. While it would take too much time to go over these picks 1 by 1,
        we can try to emerge a general pattern from the top suggestions. If we
        look at the best choices, we see they fit mostly in 2 categories:
      </p>
      <ul className="list-disc pl-6 my-4">
        <li>
          <strong>Roaming/Hook supports</strong>: Pyke, Thresh, Bard, Blitzcrank
        </li>
        <li>
          <strong>Melee Peeling tanks</strong>: Galio, Maokai, Tahm Kench Taric,
          Shen
        </li>
      </ul>

      <p>
        So why are Roaming/Hook supports and Peeling tanks good here? This is
        what we will see in the next section, seeing how{" "}
        <span className="brand-text">LoLDraftAI</span> could help with draft
        preparation.
      </p>

      <h2>Draft preparation with LoLDraftAI</h2>
      <p>
        While in soloqueue you can use{" "}
        <span className="brand-text">LoLDraftAI</span> to get champion
        recommendations, this option is not available in pro play. Instead{" "}
        <span className="brand-text">LoLDraftAI</span> can be used for draft
        preparation by filling only 1 side of the team and using champion
        suggestions. Here is what we can do to find good support picks:
      </p>

      <ClickableImage
        src="/blog/lr-vs-nord-analysis/draft-preperation-suggestions.png"
        alt="Draft preparation suggestions by LoLDraftAI"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />

      <p>
        By filling out only 1 side with the 4 champions we aim to pick, we can
        use <span className="brand-text">LoLDraftAI</span> to suggest Support
        picks, this will take into account only the team synergies since we do
        not see the enemy team. This is a strong use case to determine the best
        champion combinations. Here are the top suggested meta supports:
      </p>

      <div className="overflow-x-auto my-6">
        <table className="min-w-full border-collapse">
          <thead>
            <tr>
              <th className="border border-gray-300 px-4 py-2">Champion</th>
              <th className="border border-gray-300 px-4 py-2">Winrate</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Pyke</td>
              <td className="border border-gray-300 px-4 py-2">59.3%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Bard</td>
              <td className="border border-gray-300 px-4 py-2">58.7%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Thresh</td>
              <td className="border border-gray-300 px-4 py-2">57.7%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Sylas</td>
              <td className="border border-gray-300 px-4 py-2">57.5%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Shaco</td>
              <td className="border border-gray-300 px-4 py-2">56.7%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Vel&apos;Koz</td>
              <td className="border border-gray-300 px-4 py-2">55.7%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Neeko</td>
              <td className="border border-gray-300 px-4 py-2">55.3%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">LeBlanc</td>
              <td className="border border-gray-300 px-4 py-2">55.0%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Poppy</td>
              <td className="border border-gray-300 px-4 py-2">54.8%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Alistar</td>
              <td className="border border-gray-300 px-4 py-2">54.7%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Shen</td>
              <td className="border border-gray-300 px-4 py-2">54.4%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Galio</td>
              <td className="border border-gray-300 px-4 py-2">54.1%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Pantheon</td>
              <td className="border border-gray-300 px-4 py-2">53.6%</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">Taric</td>
              <td className="border border-gray-300 px-4 py-2">53.5%</td>
            </tr>
          </tbody>
        </table>
      </div>

      <p>
        Here again, if we filter out the solo queue supports such as Shaco,
        Vel&apos;Koz, we see the same pattern of:
      </p>
      <ul className="list-disc pl-6 my-4">
        <li>
          <strong>Roaming supports</strong>: Pyke, Bard, Thresh
        </li>
        <li>
          <strong>Melee peel supports</strong>: Poppy, Alistar, Shen, Galio,
          Taric
        </li>
      </ul>

      <p>
        So why do these work so well in this Draft? While{" "}
        <span className="brand-text">LoLDraftAI</span> and AI in general is a
        powerful tool, it cannot explain its answers, so here is my
        interpretation.
      </p>

      <h3>1) Why Roaming supports work well</h3>
      <p>The roaming supports work really well because of a few factors</p>
      <ul className="list-disc pl-6 my-4">
        <li>
          Ivern and Cho&apos;gath are good at ganking sidelanes, having an
          additional support that pairs up with them is really strong.
        </li>
        <li>
          The snowball potential of this draft is really strong, with all lanes
          being really hard to deal with once they are fed, and with
          Cho&apos;Gath objective control to avoid throws.
        </li>
        <li>Ezreal can safely farm while the support is roaming</li>
      </ul>

      <h3>2) Why Melee peel supports work well</h3>
      <p>Here are why I think melee peel supports work really well here:</p>
      <ul className="list-disc pl-6 my-4">
        <li>
          They are still capable of roams, while not as good as category 1, the
          same points apply.
        </li>
        <li>
          They pair really well with Ivern: their lack of reach is compensated
          by Ivern Q and they are super strong with Ivern E in early game
          fights.
        </li>
        <li>
          They are effective at protecting the only backline carry of this team:
          Ezreal. It&apos;s important to note that Ezreal will get focused hard,
          so having melee supports that can peel for him is important in case
          the game is even(in case of snowball, Voli and Cho can carry by
          themselves, which is pretty much what happened in the game).
        </li>
      </ul>

      <p>
        Had Los Ratones had this preparation, they surely would have considered
        these champions in draft instead of just Rell or Rakan.
      </p>

      <h2>Conclusion</h2>
      <p>
        In this article we have seen how{" "}
        <span className="brand-text">LoLDraftAI</span> can be an amazing tool
        both in solo queue but also in pro play. While in solo queue it can give
        you champion recommendations on the fly, in pro play it can still help
        prepare and analyze games. We know from voice comms that the support
        pick discussion was quickly centered around Rell or Rakan, but if Los
        Ratones had the preparation from{" "}
        <span className="brand-text">LoLDraftAI</span>, they would have
        prioritized the debate around different picks and ended up with an
        better draft.
      </p>

      <div className="mt-8 p-4 bg-primary/10 rounded-lg">
        <h2 className="text-xl font-bold mb-2">
          Ready to Improve Your Drafts?
        </h2>
        <p>
          Experience the advanced draft analysis capabilities of{" "}
          <span className="brand-text">LoLDraftAI</span> yourself:
        </p>
        <ul>
          <li>
            Use the <Link href="/draft">web version</Link> for draft analysis
          </li>
          <li>
            Download the <Link href="/download">desktop app</Link> for live
            draft tracking and champion recommendations
          </li>
        </ul>
      </div>
    </BlogLayout>
  );
}
