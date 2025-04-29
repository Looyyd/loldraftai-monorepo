import { RateLimit } from "@/app/lib/rate-limit";
import { PredictionTracking } from "@/app/lib/prediction-tracking";
import { NextResponse } from "next/server";

const backendUrl = process.env.INFERENCE_BACKEND_URL ?? "http://127.0.0.1:8000";
const proBackendUrl = process.env.PRO_INFERENCE_BACKEND_URL;
const backendApiKey = process.env.INFERENCE_BACKEND_API_KEY;

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}

export async function POST(request: Request) {
  const isAllowed = await RateLimit.checkRateLimit();
  if (!isAllowed) {
    return new NextResponse(
      JSON.stringify({ error: "Too many requests. Please try again later." }),
      {
        status: 429,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
          "Retry-After": "60",
        },
      }
    );
  }

  try {
    // Track prediction attempt (only tracks once per day per IP)
    await PredictionTracking.trackPredictionIfNeeded();

    const body = await request.json();

    // Check if using pro model (numerical_elo = -1)
    const isPro = body.numerical_elo === -1;
    if (isPro && !proBackendUrl) {
      return NextResponse.json(
        { error: "Pro model not available" },
        {
          status: 404,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Content-Type": "application/json",
          },
        }
      );
    }

    // Use pro backend URL if numerical_elo is -1 and proBackendUrl is available
    const targetUrl = isPro ? proBackendUrl! : backendUrl;
    // TODO: this could be cleaner
    // Set numerical elo to 0 for pro
    if (isPro) {
      body.numerical_elo = 0;
    }

    const response = await fetch(`${targetUrl}/predict-in-depth`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": backendApiKey || "",
      },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      if (response.status === 403) {
        console.error("API key validation failed");
        throw new Error("API key validation failed");
      }
      console.error("Detailed prediction error:", response);
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return new NextResponse(JSON.stringify(data), {
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch (error) {
    console.error("Detailed prediction error:", error);
    return new NextResponse(
      JSON.stringify({ error: "Failed to get detailed prediction" }),
      {
        status: 500,
        headers: {
          "Content-Type": "application/json",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  }
}
