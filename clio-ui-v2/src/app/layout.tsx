import type { Metadata } from "next";
import Link from "next/link";
import { AgentationInit } from "@/components/clio/agentation-init";
import { ThemeProvider } from "@/components/theme-provider";
import { ThemeToggle } from "@/components/theme-toggle";
import { Toaster } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import "./globals.css";

export const metadata: Metadata = {
  title: "CLIO Run Explorer",
  description:
    "Privacy-preserving monitoring for CLIO pipeline runs and artifacts.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased">
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          {/* TooltipProvider is required for all Tooltip components - do not remove */}
          <TooltipProvider>
            <AgentationInit />
            <header className="fixed inset-x-0 top-0 z-50 flex h-14 items-center justify-between border-b border-border/40 bg-background/60 px-3 backdrop-blur-lg sm:h-16 sm:px-6">
              <Link
                href="/"
                className="max-w-[72vw] truncate text-sm font-semibold tracking-tight text-foreground/90 hover:text-foreground sm:max-w-none"
              >
                CLIO | Anthropic&apos;s Conversational Analysis
              </Link>
              <ThemeToggle />
            </header>
            <div className="min-h-screen pt-14 pb-14 sm:pt-16 sm:pb-12">
              {children}
            </div>
            <footer className="fixed inset-x-0 bottom-0 z-40 flex h-12 items-center justify-center border-t border-border/40 bg-background/60 px-3 backdrop-blur-lg sm:px-4">
              <p className="text-center text-[11px] text-muted-foreground sm:text-xs">
                Created with{" "}
                <span aria-label="love" role="img">
                  ❤️
                </span>{" "}
                by{" "}
                <a
                  href="https://adhishthite.com/"
                  target="_blank"
                  rel="noreferrer"
                  className="font-medium text-foreground/90 hover:text-foreground"
                >
                  Adhish Thite
                </a>{" "}
                in 🇮🇳 · <em className="italic">Assisted by AI</em>
              </p>
            </footer>
          </TooltipProvider>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
