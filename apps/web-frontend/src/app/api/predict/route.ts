import { RateLimit } from "@/app/lib/rate-limit";
import { headers } from "next/headers";
import { NextResponse } from "next/server";

const backendUrl = process.env.INFERENCE_BACKEND_URL ?? "http://127.0.0.1:8000";
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
  const headersList = headers();
  const ip = (await headersList).get("x-forwarded-for") || "unknown";

  // Check rate limit
  const isAllowed = await RateLimit.checkRateLimit(ip);
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
    const body = await request.json();

    const response = await fetch(`${backendUrl}/predict`, {
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
    console.error("Prediction error:", error);
    return new NextResponse(
      JSON.stringify({ error: "Failed to get prediction" }),
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
