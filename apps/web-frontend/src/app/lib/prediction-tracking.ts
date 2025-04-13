import { kv } from "@vercel/kv";
import { track } from "@vercel/analytics/server";
import { headers } from "next/headers";

export class PredictionTracking {
  private static async shouldTrackPrediction(ip: string): Promise<boolean> {
    const now = new Date();
    const dateKey = now.toISOString().split("T")[0]; // YYYY-MM-DD format
    const trackingKey = `prediction-tracked:${ip}:${dateKey}`;

    try {
      const hasTracked = await kv.get(trackingKey);
      if (!hasTracked) {
        // Set tracking flag with 24h expiration
        await kv.set(trackingKey, true, { ex: 24 * 60 * 60 });
        return true;
      }
      return false;
    } catch (error) {
      console.error("Prediction tracking error:", error);
      return true; // Fail open - better to potentially track twice than miss tracking
    }
  }

  static async trackPredictionIfNeeded(): Promise<void> {
    const headersList = await headers();
    const ip = headersList.get("x-forwarded-for") || "unknown";

    const shouldTrack = await this.shouldTrackPrediction(ip);
    if (shouldTrack) {
      await track("Prediction Request");
    }
  }
}
