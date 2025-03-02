// /apps/web-frontend/src/app/layout.tsx
import Link from "next/link";
import { ThemeProvider } from "@/components/theme-provider";
import { Analytics } from "@vercel/analytics/react";
import { SpeedInsights } from "@vercel/speed-insights/next";
import { Chakra_Petch, Inter } from "next/font/google";
import type { Metadata, Viewport } from "next";
import "./globals.css";
import {
  NavigationMenu,
  NavigationMenuList,
  NavigationMenuItem,
  NavigationMenuLink,
  navigationMenuTriggerStyle,
} from "@draftking/ui/components/ui/navigation-menu";
import { ClarityProvider } from "@/components/clarity-provider";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@draftking/ui/components/ui/dropdown-menu";
import { Menu } from "lucide-react";

const chakraPetch = Chakra_Petch({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-chakra-petch",
});

const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
});

// Add this near your other constants
const CLARITY_PROJECT_ID = process.env.NEXT_PUBLIC_CLARITY_PROJECT_ID ?? "";

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#000000",
};

export const metadata: Metadata = {
  metadataBase: new URL("https://loldraftai.com"),
  title: {
    default: "LoLDraftAI | LoL Draft Analyzer and Helper",
    template: "%s | LoLDraftAI",
  },
  description:
    "LoLDraftAI is the most accurate League of Legends draft tool with AI-powered analysis for better draft predictions and team composition insights.",
  keywords: [
    "league of legends draft",
    "draft league of legends",
    "lol draft",
    "draft lol",
    "loldraft",
    "lol draft tool",
    "lol draft ai",
    "ai draft lol",
    "lol draft analyzer",
    "lol draft helper",
    "lol draft analysis",
    "LoLDraftAI",
  ],
  authors: [{ name: "looyyd" }],
  openGraph: {
    type: "website",
    locale: "en_US",
    url: "https://loldraftai.com",
    title: "LoLDraftAI - The Best League of Legends Draft Analysis Tool",
    description:
      "AI-powered League of Legends draft analysis tool to win your games",
    siteName: "LoLDraftAI",
    images: [
      {
        url: "https://media.loldraftai.com/og-image.png",
        width: 1200,
        height: 630,
        alt: "LoLDraftAI - League of Legends Draft Analysis Tool",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "LoLDraftAI - The Best League of Legends Draft Analysis Tool",
    description:
      "AI-powered League of Legends draft analysis tool to win your games",
    images: [
      {
        url: "https://media.loldraftai.com/og-image.png",
        width: 1200,
        height: 630,
        alt: "LoLDraftAI - League of Legends Draft Analysis Tool",
      },
    ],
  },
  icons: {
    icon: [
      { url: "https://media.loldraftai.com/favicon.ico" },
      {
        url: "https://media.loldraftai.com/favicon-16x16.png",
        sizes: "16x16",
        type: "image/png",
      },
      {
        url: "https://media.loldraftai.com/favicon-32x32.png",
        sizes: "32x32",
        type: "image/png",
      },
    ],
    apple: [
      {
        url: "https://media.loldraftai.com/apple-touch-icon.png",
        sizes: "180x180",
        type: "image/png",
      },
    ],
    other: [
      {
        rel: "icon",
        type: "image/png",
        sizes: "192x192",
        url: "https://media.loldraftai.com/logo192.png",
      },
      {
        rel: "icon",
        type: "image/png",
        sizes: "512x512",
        url: "https://media.loldraftai.com/logo512.png",
      },
    ],
  },
  alternates: {
    canonical: "./",
  },
};

export default async function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  let lastModified = "Unknown";
  let latestPatch = "Unknown";

  try {
    const response = await fetch(
      `${
        process.env.NEXT_PUBLIC_API_BASE_URL ?? "https://loldraftai.com"
      }/api/metadata`
    );

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const metadata = await response.json();
    lastModified = new Date(metadata.last_modified).toLocaleDateString(
      "en-US",
      {
        day: "numeric",
        month: "long",
      }
    );
    latestPatch = metadata.patches[metadata.patches.length - 1];
  } catch (error) {
    console.error("Failed to fetch model metadata:", error);
  }

  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${chakraPetch.variable} ${inter.variable} font-sans`}>
        <ThemeProvider>
          <ClarityProvider projectId={CLARITY_PROJECT_ID} />
          <div className="flex min-h-screen flex-col bg-background text-foreground">
            <nav className="sticky top-0 z-50 border-b border-border/40 bg-neutral-950">
              {/* Desktop Navigation */}
              <NavigationMenu className="mx-auto hidden px-4 py-3 md:block">
                <NavigationMenuList className="flex justify-center gap-10">
                  <NavigationMenuItem>
                    <Link href="/" legacyBehavior passHref>
                      <NavigationMenuLink
                        className={navigationMenuTriggerStyle()}
                      >
                        Home
                      </NavigationMenuLink>
                    </Link>
                  </NavigationMenuItem>
                  <NavigationMenuItem>
                    <Link href="/draft" legacyBehavior passHref>
                      <NavigationMenuLink
                        className={navigationMenuTriggerStyle()}
                      >
                        Draft Analysis
                      </NavigationMenuLink>
                    </Link>
                  </NavigationMenuItem>
                  <NavigationMenuItem>
                    <Link href="/download" legacyBehavior passHref>
                      <NavigationMenuLink
                        className={navigationMenuTriggerStyle()}
                      >
                        Download
                      </NavigationMenuLink>
                    </Link>
                  </NavigationMenuItem>
                  <NavigationMenuItem>
                    <Link href="/blog" legacyBehavior passHref>
                      <NavigationMenuLink
                        className={navigationMenuTriggerStyle()}
                      >
                        Blog
                      </NavigationMenuLink>
                    </Link>
                  </NavigationMenuItem>
                </NavigationMenuList>
              </NavigationMenu>

              {/* Mobile Navigation */}
              <div className="flex justify-end p-4 md:hidden">
                <DropdownMenu>
                  <DropdownMenuTrigger className="flex items-center">
                    <Menu className="h-6 w-6" />
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuItem className="text-lg" asChild>
                      <Link href="/">Home</Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem className="text-lg" asChild>
                      <Link href="/draft">Draft Analysis</Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem className="text-lg" asChild>
                      <Link href="/download">Download</Link>
                    </DropdownMenuItem>
                    <DropdownMenuItem className="text-lg" asChild>
                      <Link href="/blog">Blog</Link>
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              </div>
            </nav>

            <main className="flex-1">{children}</main>

            <footer className="border-t border-border/40 bg-neutral-950">
              <div className="container p-4 text-center text-sm text-muted-foreground mx-auto">
                <div className="mb-2">
                  Last model update: {lastModified} on patch {latestPatch}.
                  After a new patch, expect a few days of delay before an
                  update. Contact looyyd on Discord for bug reports or feature
                  requests.
                </div>
                <div className="text-xs">
                  LoLDraftAI isn&apos;t endorsed by Riot Games and doesn&apos;t
                  reflect the views or opinions of Riot Games or anyone
                  officially involved in producing or managing Riot Games
                  properties. Riot Games, and all associated properties are
                  trademarks or registered trademarks of Riot Games, Inc.
                </div>
              </div>
            </footer>
          </div>
        </ThemeProvider>
        <Analytics />
        <SpeedInsights />
      </body>
    </html>
  );
}
