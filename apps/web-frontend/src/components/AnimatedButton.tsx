import Link from "next/link";

interface AnimatedButtonProps {
  href: string;
  children: React.ReactNode;
}

export function AnimatedButton({ href, children }: AnimatedButtonProps) {
  return (
    <Link href={href}>
      <button className="relative inline-flex h-14 overflow-hidden rounded-full p-[1.5px] focus:outline-none focus:ring-2 focus:ring-primary/50 focus:ring-offset-2 focus:ring-offset-background transition-all duration-200 hover:scale-[1.02] hover:p-[2px]">
        <span className="absolute inset-[-1000%] animate-[spin_2s_linear_infinite] bg-[conic-gradient(from_90deg_at_50%_50%,hsl(220_70%_50%)_0%,hsl(160_60%_45%)_50%,hsl(220_70%_50%)_100%)]" />
        <span className="inline-flex h-full w-full cursor-pointer items-center justify-center rounded-full bg-slate-950 px-8 py-1 text-base font-medium text-white backdrop-blur-3xl">
          {children}
        </span>
      </button>
    </Link>
  );
}
