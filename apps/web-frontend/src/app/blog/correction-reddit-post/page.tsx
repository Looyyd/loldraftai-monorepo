"use client";

import Link from "next/link";
import { ClickableImage } from "@/components/ClickableImage";
import BlogLayout from "@/components/BlogLayout";

export default function CorrectionRedditPost() {
  return (
    <BlogLayout
      title="Correction: Reddit Post Accuracy Claims"
      date="April 7, 2025"
    >
      <p>
        This post presents a correction of the following{" "}
        <a href="https://www.reddit.com/r/leagueoflegends/comments/1joumtm/i_made_an_ai_model_that_predicts_62_of_ranked/">
          reddit post
        </a>
        . While the main claims and conclusions of the original post are
        correct, the accuracy claims were affected by overfitting issues.
      </p>
      <p>
        TLDR:The 2 days after the reddit, the true accuracy was 52% and the
        model was too confident, giving extreme predictions. The model is now
        fixed, and the accuracy is around 55%(as of april 7 2025), the model
        also gives more reasonable predictions. Thanks to user{" "}
        <a href="https://www.reddit.com/user/Impossible_Concert88/">
          /u/Impossible_Concert88
        </a>{" "}
        for trying to verify the accuracy claims, which led me to discover this
        bug.
      </p>
      <h2> Bug description</h2>
      <p>
        So why was the accuracy wrong? The issue came from the data collection
        process, instead of collecting data from 4 regions, it was actually only
        collecting from EUW, but marking some as coming from KR, OCE and NA.
        This basically means that almost all matches were duplicated 4 times.
      </p>
      <p>
        This is a problem, because you are supposed to seperate train data and
        test data. But here, because the original dataset was containing
        duplicates, even after splitting the data into 2, most rows were present
        both in train and in test data. When this happens, it is not possible to
        detect when the model overfits, which means that it will memorize
        matches outcomes, instead of learning general patterns.
      </p>
      <p>
        Why exactly the data from EUW mas marked as coming from KR, OCE and NA?
        This is because of a silly mistake. I created a riot API client, that
        can be passed the region when created. However, the region would default
        to EUW1 if no region was passed. And then in the data collection code, I
        forgot to specify the region.
      </p>
      <h3>Code mistakes</h3>
      <p>Here are the lines of code that caused the mistake:</p>
      <ClickableImage
        src="/blog/correction-reddit-post/code-mistakes.png"
        alt="Screenshots of the lines of code that caused the mistake"
        width={1000}
        height={1000}
      />
      <h2> Bug resolution</h2>
      <p>
        I quickly fixed the bug that deduplicates the rows, and uploaded a fixed
        version of the model on april 4, around 2 days after my reddit post.
        This new model has an accuracy of 55%. Data collection was also fixed,
        so perhaps after gathering more data from other regions, the accuracy
        will improve. I will also be rerunning experiments to see what model
        architecture works best, since previous experiments were biased.
      </p>
      <h2> Conclusion</h2>
      <p>
        I am sorry about the mistake in the reddit post, in the future I will
        create a simple script that let's anyone verify the model accuracy. I
        will also try to improve the model accuracy from 55%, but it is unkown
        what the actual ceiling is, my guess is that it is a few percent more
        than 55%, but 62% might be impossible, because draft is just a small
        part of the game.
      </p>
    </BlogLayout>
  );
}
