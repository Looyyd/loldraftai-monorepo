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

export async function GET() {
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
    const response = await fetch(`${backendUrl}/metadata`, {
      headers: {
        "X-API-Key": backendApiKey || "",
      },
    });

    if (!response.ok) {
      if (response.status === 403) {
        console.error("API key validation failed");
        throw new Error("API key validation failed");
      }
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    // Sort patches alphabetically and get only the last 3
    const filteredData = {
      ...data,
      patches: [...data.patches].sort().slice(-3),
    };

    return new NextResponse(JSON.stringify(filteredData), {
      headers: {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
      },
    });
  } catch (error) {
    console.error("Metadata fetch error:", error);
    return new NextResponse(
      JSON.stringify({ error: "Failed to get metadata" }),
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
