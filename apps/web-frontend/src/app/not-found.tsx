import Link from "next/link";
import CloudFlareImage from "@/components/CloudFlareImage";
import { Button } from "@draftking/ui/components/ui/button";

export default function NotFound() {
  return (
    <main className="flex min-h-screen w-full flex-col items-center bg-background text-foreground">
      {/* Header Section */}
      <div className="w-full bg-gradient-to-b from-primary/10 to-background py-8">
        <div className="container flex flex-col items-center justify-center gap-4 px-4">
          <h1 className="text-4xl font-bold tracking-tight text-primary text-center">
            404 Not Found
          </h1>
          <p className="text-xl text-center text-muted-foreground">
            Could not find requested resource
          </p>
        </div>
      </div>

      {/* Content Section */}
      <div className="container px-4 py-12 flex flex-col items-center">
        <CloudFlareImage
          src="/icons/confused_blitz.webp"
          alt="Confused Blitzcrank"
          width={400}
          height={400}
        />
        <Button asChild size="lg" className="mt-8">
          <Link className="text-xl" href="/draft">
            Return to draft analysis
          </Link>
        </Button>
      </div>
    </main>
  );
}
