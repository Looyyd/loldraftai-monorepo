"use client";

import { ClickableImage } from "@/components/ClickableImage";
import BlogLayout from "@/components/BlogLayout";

export default function CorrectionRedditPost() {
  return (
    <BlogLayout
      title="Bugfix and Correction of Reddit Post Accuracy Claims"
      date="April 7, 2025"
    >
      <p>
        This post presents a correction of the following{" "}
        <a href="https://www.reddit.com/r/leagueoflegends/comments/1joumtm/i_made_an_ai_model_that_predicts_62_of_ranked/">
          Reddit post
        </a>
        . While the main claims and conclusions of the original post are
        correct, the accuracy claims were affected by overfitting issues.
      </p>
      <p>
        TLDR: Two days after the Reddit post, the true accuracy was 52% and the
        model was too confident, giving extreme predictions. The model is now
        fixed, and the accuracy is around 55% (as of April 7 2025), the model
        also gives more reasonable predictions. Thanks to user{" "}
        <a href="https://www.reddit.com/user/Impossible_Concert88/">
          /u/Impossible_Concert88
        </a>{" "}
        for trying to verify the accuracy claims, which led me to discover this
        bug.
      </p>
      <h2>Bug description</h2>
      <p>
        So why was the accuracy wrong? The issue came from the data collection
        process, instead of collecting data from 4 regions, it was actually only
        collecting from EUW, but marking some as coming from KR, OCE and NA.
        This basically means that almost all matches were duplicated 4 times.
      </p>
      <p>
        This is a problem, because you are supposed to separate train data and
        test data. But here, because the original dataset was containing
        duplicates, even after splitting the data into 2, most rows were present
        both in train and in test data. When this happens, it is not possible to
        detect when the model overfits, which means that the model memorizes
        matche outcomes, instead of learning general patterns.
      </p>
      <p>
        Why exactly the data from EUW was marked as coming from KR, OCE and NA?
        This is because of a silly mistake. I created a Riot API client that can
        be passed the region when created. However, the region would default to
        EUW1 if no region was passed. And then in the data collection code, I
        forgot to specify the region. Furthermore, there was no constraint in
        the database to make sure the matchID is unique.
      </p>
      <h3>Code mistakes</h3>
      <p>Here are the lines of code that caused the mistake:</p>
      <ClickableImage
        src="/blog/correction-reddit-post/code-mistakes.png"
        alt="Screenshots of the lines of code that caused the mistake"
        width={1000}
        height={1000}
      />
      <h2>Bug resolution</h2>
      <p>
        I quickly fixed the bug that deduplicates the rows, and uploaded a fixed
        version of the model on April 4, around 2 days after my Reddit post.
        This new model has an accuracy of 55%. Data collection was also fixed,
        so perhaps after gathering more data from other regions, the accuracy
        will improve. I will also be rerunning experiments to see what model
        architecture works best, since previous experiments were biased.
      </p>
      <h2>Conclusion</h2>
      <p>
        I am sorry about the mistake in the Reddit post, in the future I will
        create a simple script that lets anyone verify the model accuracy. I
        will also try to improve the model accuracy from 55%, but it is unknown
        what the actual ceiling is, my guess is that it is a few percent more
        than 55%, but 62% might be impossible, because draft is just a small
        part of the game.
      </p>
      <h2> Further details for technical readers</h2>
      <p>
        The loss of the new model is of 0,684 for the win prediction task, which
        could seems high for a binary classification task(always guessing 50%
        win chance would lead to a loss of 0,693). But concidering that the task
        is really hard and noisy, it&apos;s hard to say. I will in the future a
        game to let users predict outcomes, to achieve a baseline to compare the
        model to.
      </p>
      <p>
        I will also try to provide more metrics and verification methods for the
        model. As an additional validation of the new model, here is a table of
        accuracy buckets. This evaluation works by splitting the validation
        predictions into buckets, based on the model prediction. Then we check
        if on average the team that is predicted to have a win chance of 60%
        actually wins 60% of the time. The results show that the model is well
        calibrated.
      </p>
      <div className="overflow-x-auto">
        <table className="min-w-full border-collapse border border-gray-300 my-4">
          <thead>
            <tr>
              <th className="border border-gray-300 px-4 py-2">Bucket</th>
              <th className="border border-gray-300 px-4 py-2">Num Samples</th>
              <th className="border border-gray-300 px-4 py-2">Accuracy</th>
              <th className="border border-gray-300 px-4 py-2">
                Expected Accuracy
              </th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="border border-gray-300 px-4 py-2">50-55%</td>
              <td className="border border-gray-300 px-4 py-2">428,768</td>
              <td className="border border-gray-300 px-4 py-2">0.524853</td>
              <td className="border border-gray-300 px-4 py-2">0.525</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">55-60%</td>
              <td className="border border-gray-300 px-4 py-2">271,010</td>
              <td className="border border-gray-300 px-4 py-2">0.573794</td>
              <td className="border border-gray-300 px-4 py-2">0.575</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">60-65%</td>
              <td className="border border-gray-300 px-4 py-2">78,046</td>
              <td className="border border-gray-300 px-4 py-2">0.635753</td>
              <td className="border border-gray-300 px-4 py-2">0.625</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">65-70%</td>
              <td className="border border-gray-300 px-4 py-2">4,807</td>
              <td className="border border-gray-300 px-4 py-2">0.712919</td>
              <td className="border border-gray-300 px-4 py-2">0.675</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">70-75%</td>
              <td className="border border-gray-300 px-4 py-2">386</td>
              <td className="border border-gray-300 px-4 py-2">0.797927</td>
              <td className="border border-gray-300 px-4 py-2">0.725</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">75-80%</td>
              <td className="border border-gray-300 px-4 py-2">87</td>
              <td className="border border-gray-300 px-4 py-2">0.816092</td>
              <td className="border border-gray-300 px-4 py-2">0.775</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">80-85%</td>
              <td className="border border-gray-300 px-4 py-2">41</td>
              <td className="border border-gray-300 px-4 py-2">0.878049</td>
              <td className="border border-gray-300 px-4 py-2">0.825</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">85-90%</td>
              <td className="border border-gray-300 px-4 py-2">32</td>
              <td className="border border-gray-300 px-4 py-2">0.937500</td>
              <td className="border border-gray-300 px-4 py-2">0.875</td>
            </tr>
            <tr>
              <td className="border border-gray-300 px-4 py-2">90-95%</td>
              <td className="border border-gray-300 px-4 py-2">2</td>
              <td className="border border-gray-300 px-4 py-2">1.000000</td>
              <td className="border border-gray-300 px-4 py-2">0.925</td>
            </tr>
          </tbody>
        </table>
      </div>
    </BlogLayout>
  );
}
