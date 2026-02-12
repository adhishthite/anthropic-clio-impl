"use client";

import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

type ThemeToggleProps = {
  className?: string;
};

export function ThemeToggle({ className }: ThemeToggleProps) {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  const isDark = mounted && resolvedTheme === "dark";
  const nextTheme = isDark ? "light" : "dark";
  const label = mounted ? `Switch to ${nextTheme} mode` : "Toggle color mode";

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          variant="outline"
          size="icon"
          className={cn(
            "relative border-border/70 bg-card/85 shadow-sm backdrop-blur-md",
            className,
          )}
          onClick={() => setTheme(nextTheme)}
          aria-label={label}
        >
          <Sun
            className={cn(
              "size-4 rotate-0 scale-100 transition-all",
              isDark && "-rotate-90 scale-0",
            )}
          />
          <Moon
            className={cn(
              "absolute size-4 rotate-90 scale-0 transition-all",
              isDark && "rotate-0 scale-100",
            )}
          />
          <span className="sr-only">{label}</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent>{label}</TooltipContent>
    </Tooltip>
  );
}
