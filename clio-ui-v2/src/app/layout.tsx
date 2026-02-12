import type { Metadata } from "next";
import Link from "next/link";
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
            <header className="fixed inset-x-0 top-0 z-50 flex h-11 items-center justify-between border-b border-border/40 bg-background/60 px-4 backdrop-blur-lg md:px-6">
              <Link
                href="/"
                className="text-sm font-semibold tracking-tight text-foreground/90 hover:text-foreground"
              >
                CLIO
              </Link>
              <ThemeToggle />
            </header>
            <div className="pt-11">{children}</div>
          </TooltipProvider>
          <Toaster />
        </ThemeProvider>
      </body>
    </html>
  );
}
