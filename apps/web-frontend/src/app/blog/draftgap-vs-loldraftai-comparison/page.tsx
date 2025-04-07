"use client";

import Link from "next/link";
import { ClickableImage } from "@/components/ClickableImage";
import BlogLayout from "@/components/BlogLayout";

export default function DraftGapComparison() {
  return (
    <BlogLayout
      title="DraftGap vs LoLDraftAI: A Detailed Comparison"
      date="March 2, 2025"
    >
      <p>
        Edit 7 April 2025 : this blog post statistical validation part has been
        removed, because part of the validation data was in the training set.
        Corrected version coming soon.
      </p>
      <p>
        When it comes to League of Legends draft analysis tools, not all are
        created equal. In this detailed comparison, we&apos;ll examine how{" "}
        <span className="brand-text">LoLDraftAI</span> stacks up against{" "}
        <a href="https://draftgap.com">DraftGap</a>, another popular draft
        analysis tool.
      </p>
      <h2>DraftGap shortcomings</h2>
      <p>
        In their own FAQ section, DraftGap explicitly acknowledges their
        limitations:
      </p>
      <div className="note border border-secondary/30">
        <p className="text-sm">
          <strong>Does DraftGap have any shortcomings?</strong> DraftGap is not
          perfect, and there are several things to keep in mind. The overall
          team comp identity is not taken into account. The synergy of duos
          within a team are used in the calculations, but the tool does not know
          about team comp identity like &apos;engage&apos; or &apos;poke&apos;.
          Damage composition is also not used in the calculation (but it is
          shown, above the team winrate), so you need to keep this in mind on
          your own. These shortcomings result from the fact that there is not
          enough data to make a perfect prediction. And we do not want to
          incorporate opinions like &apos;malphite is an engage champion&apos;
          into the tool, as using just data is the most objective way to make a
          decision.
        </p>
      </div>
      <p>
        As we&apos;ll demonstrate in this article,{" "}
        <span className="brand-text">LoLDraftAI</span> has overcome these
        limitations through its advanced machine learning approach, which can
        identify complex team compositions and their interactions. We will see
        some examples of when the statistical approach of DraftGap falls short.
      </p>
      <h3>DraftGap shortcoming example: full AP draft</h3>
      <p>
        Because DraftGap only uses champion pair statistics, it is totally
        unaware of the draft as a whole. For this reason, it will not understand
        when a draft only has AP Damage. This can be showcased by creating a
        full AP Draft with one team having the following champions from top to
        bot:
      </p>
      <ul>
        <li>Top: Vladimir</li>
        <li>Jungle: Fiddlesticks</li>
        <li>Middle: Kennen</li>
        <li>Bottom: Heimerdinger</li>
        <li>Support: Taric</li>
      </ul>
      <p>
        When inputting this draft into DraftGap, it predicts a 62.62% win
        chance. <span className="brand-text">LoLDraftAI</span> on the other
        hand, understands that this is a full AP Draft, and predicts a win
        chance of 40.4%.
      </p>
      <p>DraftGap prediction:</p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/full-ap-draftgap.png"
        alt="Full AP Draft DraftGap prediction"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <p>
        <span className="brand-text">LoLDraftAI</span> prediction:
      </p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/full-ap-loldraftai.png"
        alt="Full AP Draft LoLDraftAI prediction"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <p>
        Importantly, this not only impacts analysis but also champion
        suggestions. For example, against this full AP Draft,{" "}
        <span className="brand-text">LoLDraftAI</span> suggests Ornn as the best
        toplane champion. DraftGap, on the other hand, thinks that the team with
        Ornn top against a full AP Draft only has 40% win chance. Obviously Ornn
        would just be unkillable against a full AP Draft, this is a glaring
        example of how DraftGap&apos;s statistical approach is limited.
      </p>
      <h3>Shortcomings conclusion</h3>
      <p>
        This full AP draft just serves as a simple illustration, but it also
        will impact more nuanced situations. DraftGap will not understand:
      </p>
      <ul>
        <li>Blue vs Red side differences</li>
        <li>When a team has only one or multiple carries</li>
        <li>When a team has no CC</li>
        <li>When a team has low total damage</li>
        <li>When a team only has late game champions</li>
      </ul>
      <p>
        When you add up all these small subtleties, this just makes DraftGap not
        a very accurate tool, and this is what we will see in the next section
        that compares the accuracy of DraftGap to{" "}
        <span className="brand-text">LoLDraftAI</span>.
      </p>
      <h2>Statistical accuracy comparison</h2>
      <p>
        Edit 7 April 2025 : this blog post statistical validation part has been
        removed, because part of the validation data was in the training set.
        Corrected version coming soon.
      </p>

      <div className="mt-8 p-4 bg-primary/10 rounded-lg">
        <h2 className="text-xl font-bold mb-2">Try LoLDraftAI Today</h2>
        <p>
          Experience the advanced draft analysis capabilities of{" "}
          <span className="brand-text">LoLDraftAI</span> yourself:
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
      <h2>Appendix A: Dataset Verification</h2>
      <p>
        The results for DraftGap were obtained by using their source code
        available here:{" "}
        <a href="https://github.com/vigovlugt/draftgap">
          https://github.com/vigovlugt/draftgap
        </a>
        . All results can be manually verified by using their websites and the
        match id. Example verification for the first match of the dataset: Match
        results:{" "}
        <a href="https://www.leagueofgraphs.com/match/EUW/7298239127">
          https://www.leagueofgraphs.com/match/EUW/7298239127
        </a>
        <br />
      </p>
      <p>DraftGap prediction:</p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/verification-draftgap.png"
        alt="DraftGap prediction verification"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
      <p>
        <span className="brand-text">LoLDraftAI</span> prediction:
      </p>
      <ClickableImage
        src="/blog/draftgap-vs-loldraftai-comparison/verification-loldraftai.png"
        alt="LoLDraftAI prediction verification"
        width={800}
        height={450}
        className="my-6 rounded-lg"
      />
    </BlogLayout>
  );
}
